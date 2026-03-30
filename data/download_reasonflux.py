from huggingface_hub import hf_hub_download
import shutil

datasets = {
    "MBPP": "Gen-Verse/MBPP-ReasonFlux",
    "LiveCodeBench": "Gen-Verse/LiveCodeBench-ReasonFlux",
    "LiveBench": "Gen-Verse/LiveBench-ReasonFlux",
}

for name, repo_id in datasets.items():
    print(f"Downloading {name} from {repo_id} ...")
    cached_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=f"test/{name}.json",
    )
    shutil.copy(cached_path, f"./{name}.json")
    print(f"  -> saved to ./{name}.json")

print("Done.")
