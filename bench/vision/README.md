# Vision benchmark: a screenshot rebuilt as one HTML file

`vision_bench.py` sends a screenshot of a web page to the model with the request to rebuild it as a single HTML file,
and keeps the answer, the HTML, a rendering of it and the timings (`run*/stats.jsonl`). The write-up is in
[docs/RTX3090.md](../../docs/RTX3090.md#mode-vision).

## Sources of the screenshots

The three source screenshots (`site_*.png`) are captures of public pages, taken on 12 October 2026 and kept here only as
benchmark inputs; the files under `run*/` are the model's reproductions of them (text included) and renderings of those
reproductions. The pages, their text, logos and trademarks belong to their owners:

- `site_hn.png`: the front page of Hacker News (news.ycombinator.com), Y Combinator.
- `site_wikipedia.png`: the article « Nvidia » of the French Wikipedia (fr.wikipedia.org/wiki/Nvidia), text under
  CC BY-SA 4.0 by its contributors; the Nvidia name and logo are trademarks of NVIDIA Corporation.
- `site_python.png`: the home page of python.org, Python Software Foundation; "Python" and the Python logos are
  trademarks of the PSF.

They are not covered by this repository's MIT licence. If you are the owner of one of these pages and want a capture
removed, open an issue.
