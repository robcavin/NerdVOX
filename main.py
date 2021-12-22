# This is a sample Python script.
import torch
import pytorch3d.transforms as py3d_transforms
import torchvision.transforms
from PIL import Image
import math

def get_rays(image,F,C,R,T, device='cuda:0'):
    (H,W) = image.shape[1:3]
    y,x =torch.meshgrid(torch.linspace(0,H-1,H), torch.linspace(0,W-1,W))
    coords = torch.stack((x,y),dim=2)
    coords += 0.5
    rays_xy = (coords - C) / F
    rays = torch.cat((rays_xy, torch.ones((H, W, 1))),dim=2)

    rays_d = py3d_transforms.quaternion_apply(R,rays)
    assert((rays_d[:,:,2] > 0).all().item())

    # All rays are normalized so z=1 in world space, so
    #  that when multiplying by depth with sample uniformly
    #  in world z, not ray dir
    rays_d = rays_d / rays_d[:,:,2:]

    rays_o = T.tile((H,W,1))
    return rays_d, rays_o, image.permute(1,2,0)


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
def eval_sh_bases(coeffs : torch.Tensor, dirs : torch.Tensor):
    # https://beatthezombie.github.io/sh_post_1/
    result = torch.empty((*dirs.shape[:-1], 3, 9), dtype=dirs.dtype, device=dirs.device)
    x, y, z = dirs.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    result[..., 0] = coeffs[...,0]
    result[..., 1] = coeffs[...,1] * y.unsqueeze(-1)
    result[..., 2] = coeffs[...,2] * z.unsqueeze(-1)
    result[..., 3] = coeffs[...,3] * x.unsqueeze(-1)
    result[..., 4] = coeffs[...,4] * xy.unsqueeze(-1)
    result[..., 5] = coeffs[...,5] * yz.unsqueeze(-1)
    result[..., 6] = coeffs[...,6] * (2.0 * zz - xx - yy).unsqueeze(-1)
    result[..., 7] = coeffs[...,7] * xz.unsqueeze(-1)
    result[..., 8] = coeffs[...,8] * (xx - yy).unsqueeze(-1)

    return result


class Vox:
    def __init__(self, width, height, depth, scale):
        self.occupancy = torch.rand((width * height * depth), requires_grad=True)
        self.coeffs = torch.rand((width * height * depth, 3, 9), requires_grad=True)
        self.scale = scale
        self.width = width
        self.height = height
        self.depth = depth
        self.trilinear_offsets = torch.zeros((8, 3), dtype=torch.long)
        idx = 0
        for i in (0, 1):
            for j in (0, 1):
                for k in (0, 1):
                    self.trilinear_offsets[idx] = torch.tensor([i,j,k])
                    idx += 1

        self.trilinear_scales = torch.stack((1- self.trilinear_offsets, self.trilinear_offsets))

    def sample(self,coords, dirs):
        scaled_coords = coords * self.scale
        lower = torch.floor(scaled_coords).to(torch.long)
        delta = scaled_coords - lower
        one_minus_delta = 1 - delta
        sample_indices = lower.unsqueeze(1).expand(-1,8,-1) + self.trilinear_offsets
        flattened_indices = sample_indices[...,0] * self.width + sample_indices[...,1] * self.height + sample_indices[...,2]
        samples = self.occupancy[flattened_indices]
        sample_coeffs = self.coeffs[flattened_indices]
        scales = (self.trilinear_scales * torch.stack((delta,one_minus_delta),dim=1).unsqueeze(2)).sum(dim=1).prod(dim=2)

        occupancy = (scales * samples).sum(dim=-1)
        sh_coeffs= (scales.unsqueeze(-1).unsqueeze(-1) * sample_coeffs).sum(dim=-3)
        color = eval_sh_bases(sh_coeffs, dirs).sum(dim=-1)
        return color, occupancy


def render_samples(vox : Vox, samples, view_dirs, near=0.0, far=1.0, device='cuda:0'):

    num_z_samples = samples.shape[-2]
    flat_samples = samples.view(-1,3)
    flat_view_dirs = view_dirs.repeat(num_z_samples, 1)
    (flat_rgb, flat_occupancy) = vox.sample(flat_samples, flat_view_dirs)
    rgb = flat_rgb.view(-1,num_z_samples,3)
    occupancy = flat_occupancy.view(-1,num_z_samples)

    # # FIXME - THIS IS WRONG.  Before encoding, samples should be in a -1 to 1
    # #  range for the periodic function logic to make sense.
    # #  We don't know the boudning volume for x and y, only z
    # encoded_positions = encode(samples, 10)
    #
    # view_dirs = view_dirs / torch.norm(view_dirs, dim=-1, keepdim=True)
    # encoded_views = encode(view_dirs, 4).unsqueeze(1).expand(-1, num_z_samples, -1)
    #
    # (rgb, occupancy) = nerf(encoded_positions, encoded_views)

    # FIXME - Since the rays are not normalized, the distances t_i+1 - t_i != distance between two points
    #  Could rearrange this to save some compute - i.e. scale the sample distances for each normalized ray based on
    #  on it's direction.
    # FIXME - In the paper, dists are in z only?  Shouldn't it be energy along the ray?
    delta = (samples[:, 1:, 2:] - samples[:, :-1, 2:])
    dist = delta.norm(dim=2)

    # FIXME - Paper uses last distance to infinity.  This I guess captures any energy along the ray that
    #  wasn't caught by something else.  Does this work with my formulation?
    dist = torch.cat((dist, torch.tensor(1e10, device=device).expand(dist.shape[0], 1)), dim=-1)

    # FIXME - the first "delta"/"dist" should compare first point to point on ray at near dist.  Right now,
    #  I have one fewer distances than occupancies, and basically ignoring occupancy 0
    t = occupancy.squeeze(-1) * dist
    # accum = torch.zeros_like(t)
    # for i in range(num_z_samples - 1):  # FIXME - the -1 is because of the above issue
    #     accum[:, i:] += t[:, i, None]
    accum = torch.cumsum(t, -1)
    accum = torch.cat((torch.zeros((accum.shape[0], 1), device=device), accum[:, :-1]), -1)

    exp_accum = torch.exp(-accum)
    alpha = 1 - torch.exp(-t)

    weights = exp_accum * alpha

    # Their implementation - convert exp of sums to prod of exps
    # alpha = 1 - torch.exp(-t)
    # weights2 = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1),device=device), 1.-alpha + 1e-10], -1), -1)[:, :-1]

    c = torch.sum(weights.unsqueeze(-1) * rgb,
                  dim=1)  # FIXME - Skipping nearest sample as per above

    return c, weights


def render_rays(nerf_models, rays_d, rays_o, rand=True, near=0.0, far=1.0, num_coarse_samples=64, num_fine_samples=128,
                device='cuda:0'):

    #num_z_samples = num_coarse_samples
    #segments = torch.linspace(0,1,num_z_samples+1)[:-1]

    #distances = segments + torch.rand(num_z_samples) / num_z_samples
    #voxels = ray_bases.unsqueeze(1) + ray_dirs.unsqueeze(1).expand(-1, num_z_samples, -1) * distances.unsqueeze(-1)

    # # NOTE - repeat triggers a copy.  Kind of annoying, but expanding and flattening also triggers a copy
    # opacity, color = vox.sample(voxels.reshape(-1, 3), ray_dirs.repeat(num_z_samples, 1))

    # Linearly sample along ray, sampling from uniform distribution in each bucket
    coarse_z_samples = torch.linspace(near, far, num_coarse_samples + 1, device=device)[:-1]
    if True:  # FIXME - What should we do with random samples?
        coarse_z_samples = coarse_z_samples + torch.rand((rays_d.shape[0], num_coarse_samples), device=device) * (
                far - near) / num_coarse_samples

    samples = rays_o.unsqueeze(1) + rays_d.unsqueeze(1).expand(-1, num_coarse_samples, -1) * coarse_z_samples.unsqueeze(
        -1)

    rgb_coarse, weights_coarse = render_samples(nerf_models[0], samples, rays_d, near, far, device=device)

    # # FIXME - The samplePDF function is lifted from the nerf pytorch code.  They
    # #  also didn't include first or last samples... not sure why
    # with torch.no_grad():
    #     z_vals_mid = .5 * (coarse_z_samples[...,1:] + coarse_z_samples[...,:-1])
    #     fine_z_samples = sample_pdf(z_vals_mid, weights_coarse[...,1:-1], num_fine_samples)
    #
    # # FIXME - Should we detach the fine_z samples??  Might not differentiate thorugh
    #
    # z_samples, _ = torch.sort(torch.cat([coarse_z_samples, fine_z_samples], -1), -1)
    #
    # samples = rays_o.unsqueeze(1) + rays_d.unsqueeze(1).expand(-1, num_coarse_samples + num_fine_samples, -1) * z_samples.unsqueeze(
    #     -1)
    # rgb, weights = render_samples(nerf_models[1],samples,rays_d,near,far)

    z_samples = coarse_z_samples
    rgb = rgb_coarse
    weights = weights_coarse

    # z_samples = z_samples[:,1:] #if rand else z_samples[1:] # FIXME - Still an off-by-one thing
    depth = torch.sum(weights * z_samples, dim=1)

    return rgb_coarse, rgb, weights, depth


if __name__ == '__main__':
    import glob
    import json
    image_files = glob.glob("images/*png")
    prop_files = glob.glob("images/*json")

    image_files.sort()
    prop_files.sort()

    transforms = []
    for props in prop_files[:16] :
        with open(props) as f:
            c = json.load(f)
            t = torch.tensor([float(p) for p in c["camToWorld"].values()]).reshape((4,4))
            t[0] *= -1  # Flip x axis to convert coord systems
            transforms.append(t)

    images = []
    for image in image_files :
        with Image.open(image) as img:
            images.append(torchvision.transforms.ToTensor()(img).permute(1,2,0))


    # For synthetic data, rays are easy
    (H,W) = images[0].shape[:2]
    y,x =torch.meshgrid(torch.linspace(0,H-1,H), torch.linspace(0,W-1,W))
    coords = torch.stack((x,y),dim=2)
    coords += 0.5
    C = torch.tensor([W/2,H/2])
    F = H / 2 / math.tan(25 * math.pi / 180)
    rays_xy = (coords - C) / F
    rays = torch.cat((rays_xy, torch.ones((H, W, 1))),dim=2)
    rays = rays / rays.norm(dim=-1,keepdim=True)

    all_ray_dirs = []
    all_ray_bases = []
    all_colors = []
    all_cam_ids = []

    cam_id = 0
    for t in transforms :
        all_ray_dirs.append((t[:3,:3] @ rays.reshape(-1,3).t()).t())
        all_ray_bases.append(t[:3,3].tile((H*W,1)))
        all_colors.append(images[cam_id].reshape(-1,4))
        all_cam_ids.append(torch.full((1,H*W),cam_id))
        cam_id += 1

    all_ray_dirs = torch.cat(all_ray_dirs)
    all_ray_bases = torch.cat(all_ray_bases)
    all_colors = torch.cat(all_colors)
    all_cam_ids = torch.cat(all_cam_ids)

    all_indices = torch.tensor(range(len(all_colors)))

    # Mask out transparent pixels
    valid_indices = all_indices[all_colors[...,3] > 0]

    # Set rays to Z=1 to make math easier
    all_ray_dirs = all_ray_dirs / all_ray_dirs[...,2:]

    # for now, bounds = 0 to 1
    near = 0
    far = 1

    # far = all_ray_bases + all_ray_dirs
    # bounds_min = far.min(dim=0)[0]
    # bounds_max = far.max(dim=0)[0]

    depth = 256

    vox = Vox(256,256,256, torch.tensor([64,64,256]))

    batch_size = 4096
    maxes = []
    mins = []

    shuffled_indices = valid_indices[torch.randperm(len(valid_indices))]
    shuffled_dirs = all_ray_dirs[shuffled_indices]
    shuffled_bases = all_ray_bases[shuffled_indices]
    shuffled_colors = all_colors[shuffled_indices]


    loss_fn = torch.nn.MSELoss()
    learning_rate = 5e-4
    optimizer = torch.optim.Adam([vox.occupancy, vox.coeffs], learning_rate)

    for batch_idx in range(0, len(shuffled_colors), batch_size):
        optimizer.zero_grad()

        ray_dirs = shuffled_dirs[batch_idx: batch_idx + batch_size]
        ray_bases = shuffled_bases[batch_idx: batch_idx + batch_size]

        rgb_coarse, rgb, depth, weights = render_rays([vox], ray_dirs, ray_bases, num_coarse_samples=256, near=near, far=far, device='cpu')

        gt_color = shuffled_colors[batch_idx: batch_idx + batch_size]
        loss = loss_fn(rgb_coarse, gt_color[:,:3])
        loss.backward()
        optimizer.step()

        print(loss)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
