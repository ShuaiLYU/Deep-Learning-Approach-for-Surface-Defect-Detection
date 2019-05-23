import argparse
from agent import Agent

#默认参数(Notes : some params are disable )
DefaultParam = {
    "mode": "testing",  # 模式  {"training","testing" }
    "train_mode":"decision", #训练模式，{"segment":only train segment net,"decision": only train decision net, "total": both}
    "epochs_num": 50,
    "batch_size": 1,
    "learn_rate": 0.001,
    "momentum": 0.9,                 # 优化器参数(disable)
    "data_dir": "../Datasets/KolektorSDD",  # 数据路径
    "checkPoint_dir": "checkpoint",  # 模型保存路径
    "Log_dir": "Log",  # 日志打印路径
    "valid_ratio": 0,  # 数据集中用来验证的比例  (disable)
    "valid_frequency": 3,  # 每几个周期验证一次  (disable)
    "save_frequency": 2,  # 几个周期保存一次模型
    "max_to_keep": 10,  # 最多保存几个模型
    "b_restore": True,  # 导入参数
    "b_saveNG": True,  # 测试时是否保存错误的样本  (disable)
}

def parse_arguments():
    """
        Parse the command line arguments of the program.
    """

    parser = argparse.ArgumentParser(description='Train or test the CRNN model.')

    parser.add_argument(
        "--train_segment",
        action="store_true",
        help="Define if we wanna to train the segment net"
    )
    parser.add_argument(
        "--train_decision",
        action="store_true",
        help="Define if we wanna to train the decision net"
    )
    parser.add_argument(
        "--train_total",
        action="store_true",
        help="Define if we wanna to train the total net"
    )

    parser.add_argument(
        "--pb",
        action="store_true",
        help="Define if we wanna to get the pbmodel"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Define if we wanna test the model"
    )
    parser.add_argument(
        "--anew",
        action="store_true",
        help="Define if we try to start from scratch  instead of  loading a checkpoint file from the save folder",
    )
    parser.add_argument(
        "-vr",
        "--valid_ratio",
        type=float,
        nargs="?",
        help="How the data will be split between training and testing",
        default=DefaultParam["valid_ratio"]
    )
    parser.add_argument(
        "-ckpt",
        "--checkPoint_dir",
        type=str,
        nargs="?",
        help="The path where the pretrained model can be found or where the model will be saved",
        default=DefaultParam["checkPoint_dir"]
    )
    parser.add_argument(
        "-dd",
        "--data_dir",
        type=str,
        nargs="?",
        help="The path to the file containing the examples (training samples)",
        default=DefaultParam["data_dir"]
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        nargs="?",
        help="Size of a batch",
        default=DefaultParam["batch_size"]
    )
    parser.add_argument(
        "-en",
        "--epochs_num",
        type=int,
        nargs="?",
        help="How many iteration in training",
        default=DefaultParam["epochs_num"]
    )


    return parser.parse_args()

def main():
    """

    """
    #导入默认参数
    param=DefaultParam
    #从命令行更新参数
    args = parse_arguments()
    if not args.train_segment and   not args.train_decision and  not args.train_total and not args.test and not args.pb:
        print("If we are not training, and not testing, what is the point?")
    if  args.train_segment:
        param["mode"]="training"
        param["train_mode"] = "segment"
    if args.train_decision:
        param["mode"]="training"
        param["train_mode"] = "decision"
    if args.train_total:
        param["mode"]="training"
        param["train_mode"] = "total"
    if args.test :
        param["mode"] = "testing"
    if args.pb :
        param["mode"] = "savePb"
    if args.anew:
        param["b_restore"] =False
    param["data_dir"] = args.data_dir
    param["valid_ratio"] = args.valid_ratio
    param["batch_size"] = args.batch_size
    param["epochs_num"] = args.epochs_num
    param["checkPoint_dir"] = args.checkPoint_dir

    agent=Agent(param)
    agent.run()

if __name__ == '__main__':
    main()

