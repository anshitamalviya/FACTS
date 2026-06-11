from networks.unet import UNet


def net_factory(net_type="unet", in_chns=1, class_num=4):
    """Create the segmentation network used by FACTS."""
    if net_type != "unet":
        raise ValueError(f"Unsupported net_type '{net_type}'. This release includes only UNet.")
    return UNet(in_chns=in_chns, class_num=class_num).cuda()
