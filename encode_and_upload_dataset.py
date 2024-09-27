import tqdm
from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes, get_default_encoding
from lerobot.common.datasets.push_dataset_to_hub.aloha_hdf5_format import to_hf_dataset
from lerobot.common.datasets.utils import calculate_episode_data_index, create_branch
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.compute_stats import compute_stats
from pathlib import Path
import shutil
import torch
import glob
import os 
from lerobot.common.datasets.video_utils import encode_video_frames
from lerobot.scripts.push_dataset_to_hub import (
    push_dataset_card_to_hub,
    push_meta_data_to_hub,
    push_videos_to_hub,
    save_meta_data,
)
fps = 30
num_episodes = 20

for dir_idx in [2]:
    repo_id = f'yanivmel1/post_hack_cube_{dir_idx}'
    local_dir = f'/home/aloha/Desktop/lerobot/data/yaniv3/post_hack_cube_{dir_idx}'


    local_dir = Path(local_dir)
    episodes_dir = os.path.join(local_dir, 'episodes')
    videos_dir = os.path.join(local_dir, 'videos')
    videos_dir = Path(videos_dir)
    video = True

    image_keys = ["observation.images.laptop","observation.images.phone"]
    # Use ffmpeg to convert frames stored as png into mp4 videos
    for episode_index in tqdm.tqdm(range(num_episodes)):
        for key in image_keys:
            tmp_imgs_dir = videos_dir / f"{key}_episode_{episode_index:06d}"
            fname = f"{key}_episode_{episode_index:06d}.mp4"
            video_path = local_dir / "videos" / fname
            if video_path.exists():
                # Skip if video is already encoded. Could be the case when resuming data recording.
                continue
            # note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
            # since video encoding with ffmpeg is already using multithreading.
            encode_video_frames(tmp_imgs_dir, video_path, fps, overwrite=True)
            shutil.rmtree(tmp_imgs_dir)


    pth_files = glob.glob(episodes_dir + '/*.pth')

    ep_dicts = []
    for ep_path in pth_files:
        ep_dict = torch.load(ep_path)
        ep_dicts.append(ep_dict)
    data_dict = concatenate_episodes(ep_dicts)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)

    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": video,
    }
    if video:
        info["encoding"] = get_default_encoding()

    lerobot_dataset = LeRobotDataset.from_preloaded(
        repo_id=repo_id,
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=videos_dir,
    )
    # stats = {}
    # print("skip stats computation")
    stats = compute_stats(lerobot_dataset)

    hf_dataset = hf_dataset.with_format(None)  # to remove transforms that cant be saved
    hf_dataset.save_to_disk(str(local_dir / "train"))

    meta_data_dir = local_dir / "meta_data"
    save_meta_data(info, stats, episode_data_index, meta_data_dir)

    hf_dataset.push_to_hub(repo_id, revision="main")
    push_meta_data_to_hub(repo_id, meta_data_dir, revision="main")
    push_dataset_card_to_hub(repo_id, revision="main")
    if video:
        push_videos_to_hub(repo_id, videos_dir, revision="main")
    create_branch(repo_id, repo_type="dataset", branch=CODEBASE_VERSION)

