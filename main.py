#
#
#
import sys, getopt
from experiments import run_exp

COMMAND_HELP_STR = 'main.py \n-e <num_experiments [default=1]> \n-m <model_name [vgg, lenet, alexnet]> \n-c <epochs [default=1]>'

def main(argv):
    num_experiments = '1'  # default
    model_name = 'NONE'
    epochs = '1'  # default
    try:
        opts, args = getopt.getopt(argv, "he:m:c:", ["experiments=", "model=", "epoch="])
    except getopt.GetoptError:
        print COMMAND_HELP_STR
        sys.exit(2)
    if (opts == []):
        print COMMAND_HELP_STR
        sys.exit()

    for opt, arg in opts:
        if opt == '-h':
            print COMMAND_HELP_STR
            sys.exit()
        elif opt in ("-e", "--experiments"):
            num_experiments = arg
        elif opt in ("-m", "--model"):
            model_name = arg
        elif opt in ("-c", "--epoch"):
            epochs = arg

    #run experiment
    run_exp(model_name,epochs,num_experiments)

if __name__ == "__main__":
    main(sys.argv[1:])
