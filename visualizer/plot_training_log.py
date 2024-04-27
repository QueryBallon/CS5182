import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Plot Graph')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')

    return parser.parse_args()

def main(args):
    # Load the CSV file
    df = pd.read_csv(args.log_dir)

    # Extract the necessary columns
    epoch = df['Epoch']
    train_accuracy = df['Train Accuracy']
    test_accuracy = df['Test Accuracy']
    class_avg_miou = df['Class avg mIOU']
    inctance_avg_iou = df['Inctance avg mIOU']
    training_loss = df['Traning Loss']
    # validation_loss = df['Validation Loss']
    # start_training_time = df['Start Training Time']
    # start_testing_time = df['Start Testing Time']
    # end_time = df['End Time']

    # if start_training_time is not None and end_time is not None:
    #     st = datetime.fromtimestamp(round(start_training_time[0]))
    #     et = datetime.fromtimestamp(round(end_time[len(end_time)-1]))
    #     print(st, et)
    #     print(((end_time[len(end_time)-1]) - (start_training_time[0])) / 3600)

    # Plot the graph
    plt.figure(figsize=(10, 6))
    plt.plot(epoch, train_accuracy, label='Train Accuracy')
    plt.plot(epoch, test_accuracy, label='Test Accuracy')
    plt.plot(epoch, class_avg_miou, label='Class avg mIOU')
    plt.plot(epoch, inctance_avg_iou, label='Inctance avg mIOU')
    plt.plot(epoch, training_loss, label='Traning Loss')
    # plt.plot(epoch, validation_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    plt.title('Train and Test Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    main(args)
