"""
main file to call the explanations methods and run experiments, given a pre-trained
model and a data loader.
© copyright Tyler Lawson, Saeed khorram. https://github.com/saeed-khorram/IGOS
"""
import sys
import os
sys.path.append('/usr/local/lib/python3.7/site-packages')
import torchvision.models as models
from torch.autograd import Variable

from args import init_args
from utils import *
from methods_helper import *
from methods import IGOS, iGOS_p, iGOS_pp
from detectors.m_rcnn import m_rcnn
from detectors.f_rcnn import f_rcnn
from detectors.yolo import yolov3spp


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)


retfound_mae_dir = os.path.join(parent_dir, 'RETFound_MAE')
sys.path.append(retfound_mae_dir)
import models_vit
from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_

dinov2_dir = os.path.join(parent_dir, 'dinov2_with_attention_extraction')
sys.path.append(dinov2_dir)
from dinov2.models.vision_transformer import vit_small, vit_base, vit_large

def gen_explanations(model, dataloader, args):

    model.eval()

    out_dir = init_logger(args)

    if args.method == "I-GOS":
        method = IGOS
    elif args.method == "iGOS+":
        method = iGOS_p
    elif args.method == "iGOS++":
        method = iGOS_pp
    else:
        raise ValueError("the method does not exist. Choose from IGOS or iGOS++")

    eprint(f'Size is {args.size}x{args.size}')

    i_img = 0
    i_obj = 0
    total_del, total_ins, total_time = 0, 0, 0

    for data in dataloader:
        # unpack images and turn them into variables
        image, blur = data
        image, blur = Variable(image).cuda(), Variable(blur).cuda()

        pred_data = get_predict(image, model, args, threshold=0.2)

        if pred_data['no_res'] == True:
            eprint(f'{args.opt}-{args.method:6} ({i_img}- / {i_obj} samples) skip')
            i_img += 1
            continue

        # calculate init area
        pred_data = get_initial(pred_data, args.diverse_k, args.init_posi, 
                                args.init_val, args.input_size, args.size)

        # generate masks
        for l_i, label in enumerate(pred_data['labels']):

            # fix the proposal or use the same box for detectors
            fix_model = model_fix(model, args.model, args.model_file, pred_data, l_i, label)

            now = time.time()

            masks = method(
                model=fix_model,
                model_name=args.model,
                init_mask=pred_data['init_masks'][l_i],
                image=image.detach(),
                baseline=blur.detach(),
                label=label.unsqueeze(0),
                size=args.size,
                iterations=args.ig_iter,
                ig_iter=args.iterations,
                L1=args.L1,
                L2=args.L2,
                alpha=args.alpha,
            )

            total_time += time.time() - now

            # Calculate the scores for the masks
            del_scores, ins_scores, del_curve, ins_curve, index = metric(
                image,
                blur,
                masks.detach(),
                fix_model,
                args.model,
                label,
                l_i,
                pred_data,
                size=args.size
            )

            # # save heatmaps, images, and del/ins curves
            save_heatmaps(masks, image, args.size, i_img, l_i, out_dir, 
                          args.model, pred_data['boxes'][l_i], classes, 
                          label, out=args.input_size)
            save_curves(del_curve, ins_curve, index, i_img, l_i, out_dir)
            save_images(image, i_img, l_i, out_dir, classes, label)

            # log info
            total_del += del_scores.sum().item()
            total_ins += ins_scores.sum().item()
            i_obj += 1

            eprint(
                f'{args.opt}-{args.method:6} ({i_img}-{l_i} / {i_obj} samples)'
                f' Deletion (Avg.): {total_del / i_obj:.05f}'
                f' Insertion (Avg.): {total_ins / i_obj:.05f}'
                f' Time (Avg.): {total_time / i_obj:.03f}'
            )

        i_img += 1
        if i_img >= args.num_samples:
            break

    model.train()


if __name__ == "__main__":

    args = init_args()
    eprint(f"args:\n {args}")

    torch.manual_seed(args.manual_seed)

    init(args.input_size)
    init_sns()

    classes = get_imagenet_classes(args.dataset, args.model)

    dataset = ImageSet(args.data, image_size=args.input_size, blur=True)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=args.shuffle,
        num_workers=4
    )

    eprint("Loading the model...")

    if args.model == 'vgg19':
        model = models.vgg19(pretrained=True, progress=True).cuda()

    elif args.model == 'resnet50':
        model = models.resnet50(pretrained=True, progress=True).cuda()

    elif args.model == 'm-rcnn':
        model = m_rcnn(url = args.model_file)
        model = model.cuda()

    elif args.model == 'f-rcnn':
        model = f_rcnn(url = args.model_file)
        model = model.cuda()
    
    elif args.model == 'yolov3spp':
        model = yolov3spp(url = args.model_file)
        model = model.to('cuda')
        
    elif args.model == 'retfound':
        model = models_vit.__dict__['vit_large_patch16'](
            num_classes=2,
            drop_path_rate=0.2,
            global_pool=True,
        )

        weight_path = args.model_file
        # load RETFound weights
        checkpoint = torch.load(weight_path, map_location='cpu')
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        interpolate_pos_embed(model, checkpoint_model)
        msg = model.load_state_dict(checkpoint_model, strict=False)
        trunc_normal_(model.head.weight, std=2e-5)

        model = model.to('cuda')
    elif args.model == 'dinov2':
        model = vit_base(
                patch_size=14,
                img_size=526,
                init_values=1.0,
                num_register_tokens=n_register_tokens,
                block_chunks=0
        )
        weight_path = args.model_file
        state_dict = torch.load(weight_path)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('transformer.'):
                new_key = k[len('transformer.'):]
            else:
                new_key = k
            if not new_key.startswith('classifier.'):
                new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)
        for p in model.parameters():
            p.requires_grad = False
         model = model.to('cuda')
    else:
        raise ValueError("Model not defined.")

    for child in model.parameters():
        child.requires_grad = False

    eprint(f"Model({args.model}) successfully loaded!\n")

    gen_explanations(model, data_loader, args)

