from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import nn_model
import data_processing
import numpy as np

def main():
    
    parser = ArgumentParser("PCTADW",
                          formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    
    parser.add_argument('--input_text', nargs='?', required=True, help='Input text file.')
    
    parser.add_argument('--input_edges', nargs='?', required=True, help='Input edgelist file.')
    
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of times to iterate over each node.')
    
    parser.add_argument('--vector_dim', default=100, type=int, help='Dimension of the representation vector. (For PCTADW-1 the resulted dimension will be 2 times vector_dim.)')
    
    parser.add_argument('--alpha', default=0.0, type=float, help='Regulizer parameter.')
    
    parser.add_argument('--m', default=5, type=int, help='Maximum times each node to be fed into the neural network.')
    
    parser.add_argument('--batchsize', default=1024, type=int, help='Batch size for training.')
    
    parser.add_argument('--window_size', default=2, type=int, help='The distance within which to sample the s-parent/s-child for each node.')
    
    parser.add_argument('--training_weight_outputparent', default=1.0, type=float, help='The training weight for predicting s-parent for each node.')
    
    parser.add_argument('--training_weight_outputchild', default=1.0, type=float, help='The training weight for predicting s-child for each node.')
    
    parser.add_argument('--training_weight_outputword', default=1.0, type=float, help='The training weight for predicting the word in the text for each node.')
    
    parser.add_argument('--split_sample_size', default=50000, type=int, help='The number of samples above which to do splitting. In training the samples will be splited into several parts to solve the problem of limited memory.')
    
    parser.add_argument('--model_name', default='PCTADW-2', type=str, help='Batch size for training')
    
    parser.add_argument('--output', required=True, help='Output representation file')
    
    
    args = parser.parse_args()

    data = data_processing.file_processing(args.input_text, args.input_edges)
    
    print("Training...")

    embedding = nn_model.train(data, num_epochs=args.num_epochs, m=args.m, vector_dim=args.vector_dim, alpha=args.alpha, batchsize=args.batchsize, window_size=args.window_size, training_weight={'outputparent':args.training_weight_outputparent, 'outputword':args.training_weight_outputword, 'outputchild':args.training_weight_outputchild}, split_sample_size=args.split_sample_size, model_name=args.model_name)
    
    with open(args.output, 'w') as f:
        np.savetxt(f, embedding)


if __name__ == "__main__":
    
    main()


