import torch
import src.models.mar.misc as misc
import torch_fidelity
import shutil
import cv2
import numpy as np
import os
import time


def torch_evaluate(model, args):
    model.eval()
    num_steps = args.num_images // (args.batch_size * misc.get_world_size()) + 1
    save_folder = os.path.join(args.output_dir, "ariter{}-temp{}-{}cfg{}-image{}".format(
        args.num_iter, args.temperature, args.cfg_schedule, args.cfg, args.num_images))

    print("Save to:", save_folder)
    if misc.get_rank() == 0:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    class_num = args.class_num
    assert args.num_images % class_num == 0  # number of images per class must be the same
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    used_time = 0
    gen_img_cnt = 0

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        labels_gen = class_label_gen_world[world_size * args.batch_size * i + local_rank * args.batch_size:
                                           world_size * args.batch_size * i + (local_rank + 1) * args.batch_size]
        labels_gen = torch.Tensor(labels_gen).long().cuda()

        torch.cuda.synchronize()
        start_time = time.time()

        # generation
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # sampled_images = model.sample_official(bsz=args.batch_size, num_iter=args.num_iter, cfg=args.cfg,
                #                                      cfg_schedule=args.cfg_schedule, labels=labels_gen,
                #                                      temperature=args.temperature)

                import pdb; pdb.set_trace()
                if args.cfg != 1.0:
                    labels_gen = torch.cat([
                        labels_gen, torch.full_like(labels_gen, fill_value=-1)])
                sampled_images = model.sample(labels_gen,
                                              num_iter=args.num_iter, cfg=args.cfg, cfg_schedule=args.cfg_schedule,
                                              temperature=args.temperature, progress=False)

        # measure speed after the first generation batch
        if i >= 1:
            torch.cuda.synchronize()
            used_time += time.time() - start_time
            gen_img_cnt += args.batch_size
            print("Generating {} images takes {:.5f} seconds, {:.5f} sec per image".format(gen_img_cnt, used_time, used_time / gen_img_cnt))

        torch.distributed.barrier()
        sampled_images = sampled_images.detach().cpu()
        sampled_images = (sampled_images + 1) / 2

        # distributed save
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

    torch.distributed.barrier()
    time.sleep(10)
    if misc.get_rank() == 0:
        input2 = None
        fid_statistics_file = 'fid_stats/adm_in256_stats.npz'
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=input2,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=True,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
        # remove temporal saving folder
        shutil.rmtree(save_folder)

    torch.distributed.barrier()
    time.sleep(10)
