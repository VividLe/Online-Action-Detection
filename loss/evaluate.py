from misc import utils

all_class_name = [
    "BaseballPitch",
    "BasketballDunk",
    "Billiards",
    "CleanAndJerk",
    "CliffDiving",
    "CricketBowling",
    "CricketShot",
    "Diving",
    "FrisbeeCatch",
    "GolfSwing",
    "HammerThrow",
    "HighJump",
    "JavelinThrow",
    "LongJump",
    "PoleVault",
    "Shotput",
    "SoccerPenalty",
    "TennisSwing",
    "ThrowDiscus",
    "VolleyballSpiking"]


def Colar_evaluate(results, epoch, command, log_file):
    map, aps, cap, caps = utils.frame_level_map_n_cap(results)
    out = '[Epoch-{}] [IDU-{}] mAP: {:.4f}\n'.format(epoch, command, map)
    print(out)
    
    if log_file != '':
        with open(log_file, 'a+') as f:
            f.writelines(out)
        for i, ap in enumerate(aps):
            cls_name = all_class_name[i]
            out = '{}: {:.4f}\n'.format(cls_name, ap)
            # print(out)
            with open(log_file, 'a+') as f:
                f.writelines(out)


if __name__ == '__main__':
    pass
