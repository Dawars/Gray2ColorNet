import os
from pathlib import Path

import torch

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import numpy as np


def get_gpu_info():
    gpu_info = ""
    if torch.cuda.is_available():
        gpu_device = torch.cuda.get_device_properties(0)
        gpu_info = (f"{gpu_device.name} {round(gpu_device.total_memory / 1024 ** 3)} GB, "
                    f"Compute Capability {gpu_device.major}.{gpu_device.minor}")
    else:
        gpu_info = "None"

    return gpu_info


print(get_gpu_info())
if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    scores = []
    benchmark_root = Path(opt.dataroot)
    outputs = [
        benchmark_root / "benchmark/single_reference/gray_to_colornet/recolor_source/0/fortepan_183722_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/recolor_source/1/fortepan_250610_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/recolor_source/2/fortepan_183723_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/recolor_source/3/fortepan_251236_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/full_correspondence/0/fortepan_201867_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/full_correspondence/1/fortepan_229825_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/full_correspondence/2/fortepan_102400_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/full_correspondence/3/fortepan_201867_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/full_correspondence/4/fortepan_229825_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/full_correspondence/5/fortepan_102400_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/partial_reference/0/fortepan_201867_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/partial_reference/1/fortepan_229825_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/partial_reference/2/fortepan_102400_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/partial_reference/3/fortepan_201867_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/partial_reference/4/fortepan_229825_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/partial_reference/5/fortepan_102400_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/partial_source/0/fortepan_18476_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/partial_source/1/fortepan_79821_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/partial_source/2/fortepan_67270_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/partial_source/3/fortepan_18476_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/partial_source/4/fortepan_79821_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/partial_source/5/fortepan_67270_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/semantic_correspondence_strong/0/fortepan_251148_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/semantic_correspondence_strong/1/fortepan_97196_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/semantic_correspondence_strong/2/fortepan_97191_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/semantic_correspondence_strong/3/fortepan_251148_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/semantic_correspondence_strong/4/fortepan_97196_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/semantic_correspondence_strong/5/fortepan_97191_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/semantic_correspondence_weak/0/fortepan_148611_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/semantic_correspondence_weak/1/fortepan_84203_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/semantic_correspondence_weak/2/fortepan_84203_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/semantic_correspondence_weak/3/fortepan_148611_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/semantic_correspondence_weak/4/fortepan_84203_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/semantic_correspondence_weak/5/fortepan_84203_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/distractors/0/fortepan_18098_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/distractors/1/fortepan_276876_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/distractors/2/fortepan_40115_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/random_noise/0/fortepan_18098_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/random_noise/1/fortepan_276876_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/random_noise/2/fortepan_40115_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/random_noise/3/fortepan_201867_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/random_noise/4/fortepan_229825_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/random_noise/5/fortepan_102400_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/gray/0/fortepan_18098_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/gray/1/fortepan_276876_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/gray/2/fortepan_40115_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/gray/3/fortepan_201867_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/gray/4/fortepan_229825_color.jpg",
        benchmark_root / "benchmark/single_reference/gray_to_colornet/gray/5/fortepan_102400_color.jpg"]

    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        with torch.no_grad():
            model.set_input(data)
            model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        metrics = model.compute_scores()
        scores.extend(metrics)
        Path(str(outputs[i])).parent.mkdir(exist_ok=True, parents=True)
        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, str(outputs[i]), aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()
    print('Histogram Intersection: %.4f' % np.mean(scores))
