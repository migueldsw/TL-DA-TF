#
#
#
import sys, getopt


def main(argv):
    num_experiments = ''
    model_name = ''
    try:
        opts, args = getopt.getopt(argv, "he:m:", ["experiments=", "model="])
    except getopt.GetoptError:
        print 'main.py -e <experiments> -m <model name [vgg, lenet, alexnet]>'
        sys.exit(2)
    if (opts == []):
        print 'main.py -e <experiments> -m <model name [vgg, lenet, alexnet]>'
        sys.exit()

    for opt, arg in opts:
        if opt == '-h':
            print 'main.py -i <num_experiments> -o <model_name>'
            sys.exit()
        elif opt in ("-e", "--experiments"):
            num_experiments = arg
        elif opt in ("-m", "--model"):
            model_name = arg
    print 'Number of experiments is: ', num_experiments
    print 'Selected model is: ', model_name


if __name__ == "__main__":
    main(sys.argv[1:])
