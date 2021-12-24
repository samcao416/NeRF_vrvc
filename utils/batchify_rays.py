import torch


def batchify_ray(model, rays, radii, lossmult, bboxes = None, near_far=None, chuncks = 1024*7):
    N = rays.size(0)
    if N <chuncks:
        return model(rays, radii, lossmult, near_far=near_far)
    else:
        rays = rays.split(chuncks, dim=0)
        if bboxes is not None:
            bboxes = bboxes.split(chuncks, dim=0)
        else:
            bboxes = [None]*len(rays)
        if near_far is not None:
            near_far = near_far.split(chuncks, dim=0)
        else:
            near_far = [None]*len(rays)

        colors = [[],[]]
        depths = [[],[]]
        accs = [[],[]]

        ray_masks = []

        for i in range(len(rays)):
            results = model(rays[i], radii[i], lossmult[i], bboxes[i], near_far = near_far[i])
            colors[0].append(results['coarse_color'])
            depths[0].append(results['coarse_depth'])
            accs[0].append(results['coarse_acc'])

            colors[1].append(results['fine_color'])
            depths[1].append(results['fine_depth'])
            accs[1].append(results['fine_acc'])
            if results['ray_mask'] is not None:
                ray_masks.append(results['ray_mask'])

        colors[0] = torch.cat(colors[0], dim=0)
        depths[0] = torch.cat(depths[0], dim=0)
        accs[0] = torch.cat(accs[0], dim=0)

        colors[1] = torch.cat(colors[1], dim=0)
        depths[1] = torch.cat(depths[1], dim=0)
        accs[1] = torch.cat(accs[1], dim=0)
        if len(ray_masks)>0:
            ray_masks = torch.cat(ray_masks, dim=0)

        results_all = {}
        results_all['coarse_color'] = colors[0]
        results_all['coarse_depth'] = depths[0]
        results_all['coarse_acc'] = accs[0]
        results_all['fine_color'] = colors[1]
        results_all['fine_depth'] = depths[1]
        results_all['fine_acc'] = accs[1]
        results_all['ray_mask'] = ray_masks
        
        return results_all