#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from src.main import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Shadow Removal Model')
    
    # Basic parameters
    parser.add_argument('--manual_seed', type=int, default=42, help='manual seed for reproducibility')
    parser.add_argument('--devices', nargs='+', default=['0'], help='device ids')
    
    # Directory parameters
    parser.add_argument('--data_dir', nargs='+', required=True, help='path to ISTD dataset')
    parser.add_argument('--weights', default='weights', help='path to save weights')
    parser.add_argument('--logs', default='logs', help='path to save logs')
    
    # Loading parameters
    parser.add_argument('--load_args', default=None, help='path to load arguments from')
    parser.add_argument('--load_checkpoint', default=None, help='path to load checkpoint from')
    parser.add_argument('--load_weights_g1', default=None, help='path to load G1 weights from')
    parser.add_argument('--load_weights_g2', default=None, help='path to load G2 weights from')
    parser.add_argument('--load_weights_d1', default=None, help='path to load D1 weights from')
    parser.add_argument('--load_weights_d2', default=None, help='path to load D2 weights from')
    
    # Model parameters
    parser.add_argument('--net_G', default='stcgan', help='generator architecture')
    parser.add_argument('--net_D', default='stcgan', help='discriminator architecture')
    parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
    parser.add_argument('--activation', type=str, default='relu', help='activation function [relu/lrelu/selu]')
    parser.add_argument('--SELU', type=bool, default=False, help='use SELU activation')
    parser.add_argument('--NN_upconv', type=bool, default=False, help='use nearest neighbor upconvolution')
    parser.add_argument('--droprate', type=float, default=0.0, help='dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--lr_G', type=float, default=0.0002, help='generator learning rate')
    parser.add_argument('--lr_D', type=float, default=0.0002, help='discriminator learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam optimizer')
    parser.add_argument('--decay', type=float, default=0.001, help='learning rate decay per epoch')
    
    # Loss parameters
    parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
    parser.add_argument('--gan_mode', type=str, default='lsgan', help='gan mode [sgan/lsgan/rpgan/ragan]')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.weights, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)
    
    # Add tasks parameter
    args.tasks = ['train']
    
    # Run training
    main(args)
