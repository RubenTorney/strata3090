"""Local library, security boundary and full mock benchmark workflow (no GPU downloads)."""
import copy
import json
import tempfile
import time
import unittest
import urllib.request
import urllib.error
from pathlib import Path
from unittest.mock import patch
from serve.community import Community, Library, bundle, settings, replace_args
from serve.server import ByteTokenizer, MockEngine, Service, serve, ROOT
from serve.frontend import ChatTemplate


def example():
    return {"schema":1,"title":"Balanced","model":"test","rig":{"gpu":"RTX 3090","os":"Linux","ram_gib":64},
            "settings":{"--pcie-frac":.35,"--spec":4},"author":""}


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.db=Library(Path(self.tmp.name)/"community.sqlite")

    def test_roundtrip_duplicate_and_private_preferences(self):
        c=example();c['published']='2026-10-15T10:00:00+00:00'
        cid=self.db.add(c);self.assertEqual(cid,self.db.add(c))
        self.db.preference(cid,True,5)
        self.assertEqual(self.db.get(cid)['rating'],5)
        exported=self.db.export(cid)
        self.assertNotIn('rating',exported);self.assertNotIn('favorite',exported)
        self.assertEqual(exported['author'],'Anonymous')
        self.assertEqual(cid,self.db.add(exported))
        self.assertGreater(len(self.db.gpus()),25)
        self.assertEqual(Library(self.db.path).get(cid)['rating'],5)

    def test_browser_numeric_roundtrip_deduplicates(self):
        c=example();c['published']='2026-10-15T10:00:00+00:00';c['rig']['vram_gib']=24.0
        cid=self.db.add(c)
        doc=self.db.export(cid)
        doc['rig']['gpus'][0]['vram_gib']=24
        doc['rig']['vram_gib']=24
        self.assertEqual(self.db.add(doc),cid)

    def test_reject_sensitive_and_unbounded_input(self):
        for key in ('exe','args','env','api_key','cwd','before_load'):
            c=example();c[key]='bad'
            with self.assertRaises(ValueError):bundle(c)
        for v in (float('nan'),float('inf'),True,-1,2,'0.5'):
            with self.assertRaises(ValueError):settings({'--pcie-frac':v})
        with self.assertRaises(ValueError):settings({'--spec':1.5})
        with self.assertRaises(ValueError):settings({'--model':'/tmp/file'})
        with self.assertRaises(ValueError):bundle({**example(),'published':'yesterday'})
        with self.assertRaises(ValueError):bundle({**example(),'results':[{'protocol':'invented'}]})

    def test_replace_only_whitelisted_values_and_sql_injection(self):
        self.assertEqual(replace_args(['--model','/trusted','--spec','3','--vision'],{'--spec':4}),
                         ['--model','/trusted','--vision','--spec','4'])
        with self.assertRaises(ValueError):self.db.get("' OR 1=1 --")
        cid=self.db.add(example())
        with self.assertRaises(ValueError):self.db.preference(cid,True,6)


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        tok=ByteTokenizer();eng=MockEngine(tok,'def answer(): return 42\n',delay_s=.002)
        self.svc=Service(eng,tok,ChatTemplate(ROOT/'serve/chat_template.jinja'),model_name='test')
        self.svc.community=Community(self.svc,Path(self.tmp.name)/'library.sqlite')
        self.http=serve(self.svc,port=0);self.addCleanup(self.http.server_close);self.addCleanup(self.http.shutdown)
        self.base=f'http://127.0.0.1:{self.http.server_address[1]}'

    def request(self,path,data=None,headers=None):
        h={'Content-Type':'application/json',**(headers or {})}
        req=urllib.request.Request(self.base+path,data=json.dumps(data).encode() if data is not None else None,headers=h)
        try:
            with urllib.request.urlopen(req,timeout=10) as r:return r.status,json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:return e.code,json.loads(e.read())

    def wait(self):
        end=time.monotonic()+8
        while time.monotonic()<end:
            s=self.request('/community/state')[1]
            if s['job']['state']!='running':return s
            time.sleep(.02)
        self.fail('job never finished')

    def test_complete_share_import_benchmark_compare(self):
        code,c=self.request('/community/save',{'title':'Local','author':'tester','settings':{'--spec':4}})
        self.assertEqual(code,200)
        cid=c['id']
        self.assertEqual(self.request('/community/benchmark',{'id':cid})[0],200)
        s=self.wait();self.assertEqual(s['job']['state'],'done',s['job'])
        run=s['runs'][0];self.assertEqual(len(run['baseline']),2);self.assertEqual(len(run['candidate']),2)
        self.assertTrue(all(r['simulated'] for r in run['candidate']))
        code,export=self.request('/community/export',{'id':cid,'run_id':run['id']})
        self.assertEqual(code,200);self.assertEqual(export['author'],'Anonymous')
        self.assertEqual(len(export['results']),2)
        code,imported=self.request('/community/import',export)
        self.assertEqual(code,200)
        item=self.svc.community.library.get(imported['id'])
        self.assertEqual(item['origin'],'imported')
        self.assertEqual(self.request('/community/preference',{'id':imported['id'],'favorite':True,'rating':4})[0],200)
        self.assertEqual(self.request('/community/adopt',{'id':cid})[0],200)
        self.assertEqual(self.wait()['job']['state'],'done')

    def test_auth_csrf_limits_and_gate(self):
        self.assertEqual(self.request('/community/import',example(),{'Origin':'https://evil.example'})[0],403)
        self.assertEqual(self.request('/community/import',example(),{'Content-Type':'text/plain'})[0],415)
        self.assertEqual(self.request('/community/import',{'x':'a'*140000})[0],413)
        self.svc.api_key='test-key'
        self.assertEqual(self.request('/community/state')[0],401)
        self.assertEqual(self.request('/community/import',example())[0],401)
        self.svc.api_key=''
        self.svc.community.gate.acquire()
        try:
            self.assertEqual(self.request('/v1/chat/completions',{})[0],409)
            self.assertEqual(self.request('/community/state')[0],200)
        finally:self.svc.community.gate.release()

    def test_failure_restores_and_cancel_does_not_save(self):
        cid=self.svc.community.library.add(example())
        with patch.object(self.svc.community,'measure',side_effect=RuntimeError('injected')), patch.object(self.svc.community,'restart') as restart:
            self.svc.community.start('benchmark',cid)
            s=self.wait();self.assertEqual(s['job']['state'],'failed')
            self.assertEqual(restart.call_count,2)
            self.assertEqual(s['runs'],[])
        self.assertTrue(self.svc.community.gate.acquire(False));self.svc.community.gate.release()
        self.svc.engine.delay=.02
        self.svc.community.start('benchmark',cid)
        self.request('/community/cancel',{})
        self.assertEqual(self.wait()['job']['state'],'failed')
        self.assertEqual(self.svc.community.library.runs(),[])

    def test_wrong_model_and_default_disabled(self):
        cid=self.svc.community.library.add({**example(),'model':'different'})
        self.assertEqual(self.request('/community/benchmark',{'id':cid})[0],400)
        del self.svc.community
        self.assertEqual(self.request('/community/state')[0],404)
        self.assertFalse(self.request('/health')[1].get('community',False))



class MultiGpuTests(unittest.TestCase):
    def test_mixed_gpu_and_ddr_roundtrip(self):
        from serve.community import validate_rig
        r=validate_rig({'gpus':[{'name':'RTX 3090','vram_gib':24},{'name':'Tesla V100 PCIe 16GB','vram_gib':16}],
                        'os':'Linux','ram_gib':128,'ram_type':'DDR5'})
        self.assertEqual(r['vram_gib'],40)
        self.assertEqual(len(r['gpus']),2)
        self.assertEqual(bundle({**example(),'rig':r})['rig'],r)
        for invalid in ({'gpus':[]},{'gpus':[{'name':'bad','vram_gib':-1}]},{'ram_type':'made up'}):
            with self.assertRaises(ValueError):validate_rig(invalid)
        self.assertEqual(validate_rig({'gpu':'RTX 3090','vram_gib':24})['ram_type'],'Unknown')

    def test_partial_preferences_do_not_clobber(self):
        with tempfile.TemporaryDirectory() as tmp:
            db=Library(Path(tmp)/'db.sqlite');cid=db.add(example())
            db.preference(cid,True);db.preference(cid,rating=4)
            self.assertEqual(db.get(cid)['favorite'],1)
            self.assertEqual(db.get(cid)['rating'],4)

class ClusterTests(unittest.TestCase):
    def test_unified_cluster_keeps_memory_per_node(self):
        from serve.community import validate_rig
        node={'platform':'DGX Spark','count':2,'ram_gib':128,'ram_type':'LPDDR5X',
              'memory_architecture':'Unified','gpus':[{'name':'NVIDIA GB10','vram_gib':None}]}
        rig=validate_rig({'nodes':[node],'interconnect':'ConnectX-7 · 200 Gb/s'})
        self.assertEqual(rig['nodes'][0]['ram_gib'],128)
        self.assertEqual(rig['nodes'][0]['count'],2)
        self.assertIsNone(rig['nodes'][0]['vram_gib'])
        self.assertEqual(bundle({**example(),'rig':rig})['rig'],rig)
        with self.assertRaises(ValueError):validate_rig({'nodes':[{**node,'host':'ssh://evil'}]})
        with self.assertRaises(ValueError):validate_rig({'nodes':[{**node,'nodes':[node]}]})
        with self.assertRaises(ValueError):validate_rig({'nodes':[{**node,'count':0}]})

class RestartTests(unittest.TestCase):
    def test_adopt_failure_rollback_and_restore(self):
        class FakeSpawnEngine(MockEngine):
            def __init__(self,tok):
                super().__init__(tok,'answer')
                self.spawn=('/trusted/engine',['--model','/trusted/model','--prompt-cache','6'],'.',None,{},False,None)
                self.fail=False
            def restart(self,tries=1):
                if self.fail and self.spawn[1][-1]=='2':raise RuntimeError('injected start failure')
        with tempfile.TemporaryDirectory() as tmp:
            tok=ByteTokenizer();engine=FakeSpawnEngine(tok)
            svc=Service(engine,tok,ChatTemplate(ROOT/'serve/chat_template.jinja'),model_name='test')
            c=Community(svc,Path(tmp)/'db.sqlite')
            cid=c.library.add({**example(),'settings':{'--prompt-cache':2}})
            original=copy.deepcopy(engine.spawn)
            def wait():
                for _ in range(100):
                    if c.job['state']!='running':return
                    time.sleep(.01)
                self.fail('job did not finish')
            c.start('adopt',cid);wait()
            self.assertEqual(c.current()['--prompt-cache'],2)
            self.assertEqual(c.previous_spawn,original)
            c.start('restore',None);wait()
            self.assertEqual(engine.spawn,original);self.assertIsNone(c.previous_spawn)
            engine.fail=True
            c.start('adopt',cid);wait()
            self.assertEqual(c.job['state'],'failed');self.assertEqual(engine.spawn,original)
            self.assertTrue(c.gate.acquire(False));c.gate.release()

    def test_sampled_memory_per_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok=ByteTokenizer();engine=MockEngine(tok,'some code',delay_s=.002)
            svc=Service(engine,tok,ChatTemplate(ROOT/'serve/chat_template.jinja'),model_name='test')
            c=Community(svc,Path(tmp)/'db.sqlite')
            with patch.object(svc,'metrics',return_value={'hardware':{'ram_used':32*2**30,'gpu_mem_used':20*2**30,
                    'gpus':[{'index':0,'mem_used':12*2**30},{'index':2,'mem_used':8*2**30}]}}):
                result=c.measure('cold')
            self.assertEqual(result['ram_peak_gib'],32)
            self.assertEqual(result['gpu_peaks'],[{'index':0,'peak_gib':12},{'index':2,'peak_gib':8}])

class AutoMetadataTests(HttpTests):
    def test_auto_snapshot_and_metadata_whitelist(self):
        self.svc.engine.info = {"version":"0.1.44", "kv":"int8", "api_key":"not-exported", "model_path":"/private/model"}
        code, item = self.request('/community/save', {'title':'Automatic'})
        self.assertEqual(code,200)
        doc=self.svc.community.library.export(item['id'])
        self.assertEqual(doc['model'],'test')
        self.assertEqual(doc['runtime']['version'],'0.1.44')
        self.assertIn('gpus',doc['rig'])
        self.assertNotIn('api_key',doc['runtime'])
        self.assertNotIn('model_path',doc['runtime'])
        self.assertEqual(bundle(doc),doc)
        with self.assertRaises(ValueError):bundle({**doc,'runtime':{'api_key':'x'}})
        with self.assertRaises(ValueError):bundle({**doc,'runtime':{'max_context':float('nan')}})


class WorkbenchTests(unittest.TestCase):
    setUp = HttpTests.setUp
    request = HttpTests.request
    wait = HttpTests.wait

    def test_repository_scenarios_execute_and_export_their_actual_requests(self):
        from tools.strata_bench import Suite
        self.svc.engine.delay = 0
        self.svc.engine.max_context = 200000  # byte tokenizer expands the agent's document fixtures
        cid = self.svc.community.library.add(example())
        for name, count in (("chat",3),("context",2),("agent",6)):
            with self.subTest(name=name), patch('tools.strata_bench.chat',side_effect=AssertionError('no HTTP loopback')):
                self.assertEqual(self.request('/community/benchmark',{'id':cid,'benchmark':name})[0],200)
                s=self.wait();self.assertEqual(s['job']['state'],'done',s['job'])
                run=s['runs'][0]
                self.assertEqual(run['benchmark'],name)
                self.assertEqual(len(run['baseline']),count);self.assertEqual(len(run['candidate']),count)
                self.assertEqual(s['job']['completed'],count*2)
                self.assertTrue(all(r['simulated'] for r in run['candidate']))
                self.assertTrue(all(r['prompt_tokens']>0 for r in run['candidate']))
                self.assertTrue(all(a['answer_sha256']==b['answer_sha256'] for a,b in zip(run['baseline'],run['candidate'])))
                code,doc=self.request('/community/export',{'id':cid,'run_id':run['id']})
                self.assertEqual(code,200);self.assertEqual(len(bundle(doc)['results']),count)
        # Existing CLI callers retain their original HTTP transport.
        from tools import strata_bench
        self.assertIs(Suite('', '', 32768, 3, False).request,strata_bench.chat)

    def test_pause_resume_cancel_and_mutation_gate(self):
        c=self.svc.community;cid=c.library.add(example())
        self.svc.engine.delay=.015
        self.assertEqual(self.request('/community/pause',{})[0],400)
        self.assertEqual(self.request('/community/benchmark',{'id':cid})[0],200)
        self.assertEqual(self.request('/community/pause',{})[0],200)
        end=time.monotonic()+3
        while time.monotonic()<end and not c.job.get('paused'):time.sleep(.02)
        self.assertTrue(c.job['paused'])
        self.assertEqual(self.request('/community/reload',{})[0],400)
        self.assertEqual(self.request('/v1/chat/completions',{})[0],409)
        self.assertEqual(self.request('/community/resume',{})[0],200)
        self.assertEqual(self.wait()['job']['state'],'done')
        count=len(c.library.runs())
        self.assertEqual(self.request('/community/benchmark',{'id':cid})[0],200)
        self.request('/community/pause',{})
        self.assertEqual(self.request('/community/cancel',{})[0],200)
        self.assertEqual(self.wait()['job']['state'],'failed')
        self.assertEqual(len(c.library.runs()),count)
        self.assertTrue(c.gate.acquire(False));c.gate.release()

    def test_catalog_is_bounded_and_context_is_checked_before_restart(self):
        c=self.svc.community;cid=c.library.add(example())
        with patch.object(c,'restart') as restart:
            for value in ('../../tools/private.py','shell',{},'all'):
                self.assertEqual(self.request('/community/benchmark',{'id':cid,'benchmark':value})[0],400)
            self.svc.engine.max_context=4096
            for name in ('context','agent'):
                self.assertEqual(self.request('/community/benchmark',{'id':cid,'benchmark':name})[0],400)
            restart.assert_not_called()
        self.assertEqual(self.request('/community/reload',{})[0],200)
        self.assertEqual(self.wait()['job']['state'],'done')

    def test_undo_returns_to_immediately_previous_configuration(self):
        c=self.svc.community
        self.svc.engine.spawn=('/trusted/engine',['--spec','4'],'.',None,{},False,None)
        a=c.library.add({**example(),'title':'A','settings':{'--spec':2}})
        b=c.library.add({**example(),'title':'B','settings':{'--spec':3}})
        with patch.object(self.svc.engine,'restart',create=True):
            for cid in (a,b):
                c.start('adopt',cid);self.assertEqual(self.wait()['job']['state'],'done')
            self.assertEqual(c.current()['--spec'],3)
            c.start('reload',None);self.wait();self.assertEqual(c.current()['--spec'],3)
            c.start('restore',None);self.wait()
            self.assertEqual(c.current()['--spec'],2);self.assertEqual(c.active_id,a)
            self.assertFalse(c.state()['can_restore'])

if __name__=='__main__':unittest.main()
