# Hardware reference sources

Checked 2026-10-15. Curated catalog, not an exhaustive GPU inventory or a compatibility list.
The JSON seed stores stable IDs, variant names, vendor, architecture, dedicated VRAM (nullable),
source URL and verification date. Unified-memory entries use a distinct list of memory options.
No performance ranking is inferred from gaming benchmarks, TFLOPS or memory capacity.

| Family | Primary source | Scope |
|---|---|---|
| GeForce RTX 30/40/50 desktop | [NVIDIA comparison](https://www.nvidia.com/en-gb/geforce/graphics-cards/compare/) | Architecture and memory variants; laptop cards deliberately excluded |
| Tesla V100 | [NVIDIA application guide, final specifications page](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/v100-application-performance-guide.pdf) | PCIe / SXM2, 16 / 32 GB distinct; TV100 accepted as a search alias |
| Radeon RX 7000/9000 | [AMD ROCm 7.0.2 architecture reference](https://rocm.docs.amd.com/en/docs-7.0.2/reference/gpu-arch-specs.html) | Architecture and VRAM; ROCm listing does not prove Strata compatibility |
| Intel Arc A770 | [Intel specifications](https://www.intel.com/content/www/us/en/products/sku/229151/intel-arc-a770-graphics-16gb/specifications.html) | Explicit 16 GB variant |
| Intel Arc B580/B570 | [Intel B-series reference guide](https://cdrdv2-public.intel.com/839907/Intel%20Arc%20B-Series%20Graphics%20Quick%20Reference%20Guide%20V1.1.pdf) | 12 / 10 GB variants |
| Mac mini M4 / M4 Pro | [Apple 2024 technical specifications](https://support.apple.com/en-gb/121555) | Unified-memory options; DDR generation intentionally Unknown (not specified there) |
| DGX Spark / GB10 | [NVIDIA hardware guide](https://docs.nvidia.com/dgx/dgx-spark/hardware.html) | 128 GB LPDDR5x unified memory, not a separate 128 GB VRAM allocation |
| DGX Spark clusters | [NVIDIA ConnectX-7 guide](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html) | Descriptive interconnect metadata; 200 Gb/s port capability is not measured throughput |

The UI uses binary GiB for telemetry. GPU vendors label nominal memory configurations in GB;
reference capacity labels are mapped to the conventional nominal GPU capacity. A real run keeps
its measured OS/device memory separately. Unknowns stay null/Unknown. Memory bandwidth, PCIe link
training, power limits, driver versions and thermal state are not inferred by this V0 catalog.
