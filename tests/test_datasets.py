# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 18:20:32 2023

@author: WET2RNG
"""
import pytest
import torch
import pandas as pd
import numpy as np
from softsensor.datasets import SlidingWindow, batch_rec_SW


@pytest.fixture()
def df():
    d = {'in_col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
         'in_col2': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
         'in_dummy': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         'out_col1': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
         'out_col2': [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
         'out_col1_precomp': [111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123],
         'out_col2_precomp': [121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133],
         'out_dummy': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}

    df = pd.DataFrame(d)

    return df


'''
Testing of sliding window
'''


def test_SlidingWindow_len1(df):
    windowsize = 5
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    sw = SlidingWindow(df, windowsize, output_col, input_col)
    assert sw.__len__() == 9

    for i in range(sw.__len__()):
        assert sw.__getitem__(i)[0].shape == torch.Size([2, 5])
        assert sw.__getitem__(i)[1].shape == torch.Size([2, 1])


def test_SlidingWindow_len2(df):
    windowsize = 5
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    sw = SlidingWindow(df, windowsize, output_col, input_col, rnn_window=1)
    assert sw.__len__() == 13
    
    for i in range(sw.__len__()):
        assert sw.__getitem__(i)[0][0].shape == torch.Size([2, 5])
        assert sw.__getitem__(i)[0][1].shape == torch.Size([2, 1])
        assert sw.__getitem__(i)[1].shape == torch.Size([2, 1])

def test_SlidingWindow_len3(df):
    windowsize = 5
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    sw = SlidingWindow(df, windowsize, output_col, input_col, forecast=2)
    assert sw.__len__() == 9

    for i in range(sw.__len__()):
        assert sw.__getitem__(i)[0].shape == torch.Size([2, 5])
        assert sw.__getitem__(i)[1].shape == torch.Size([2, 2])

def test_SlidingWindow_len4(df):
    windowsize = 5
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    sw = SlidingWindow(df, windowsize, output_col, input_col, rnn_window=1, 
                       forecast=3)
    assert sw.__len__() == 13

    for i in range(sw.__len__()):
        assert sw.__getitem__(i)[0][0].shape == torch.Size([2, 5])
        assert sw.__getitem__(i)[0][1].shape == torch.Size([2, 1])
        assert sw.__getitem__(i)[1].shape == torch.Size([2, 3])


def test_SlidingWindow_len5(df):
    windowsize = 5
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    sw = SlidingWindow(df, windowsize, output_col, input_col, rnn_window=5, 
                       forecast=3)
    assert sw.__len__() == 13

    for i in range(sw.__len__()):
        assert sw.__getitem__(i)[0][0].shape == torch.Size([2, 5])
        assert sw.__getitem__(i)[0][1].shape == torch.Size([2, 5])
        assert sw.__getitem__(i)[1].shape == torch.Size([2, 3])

    sw = SlidingWindow(df, windowsize, output_col, input_col, rnn_window=5, 
                       forecast=3, pre_comp=['out_col1_precomp', 'out_col2_precomp'])
    assert sw.__len__() == 13

    for i in range(sw.__len__()):
        assert sw.__getitem__(i)[0][0].shape == torch.Size([2, 5])
        assert sw.__getitem__(i)[0][1].shape == torch.Size([2, 5])
        assert sw.__getitem__(i)[1].shape == torch.Size([2, 3])

def test_SlidingWindow_len6(df):
    windowsize = 5
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    sw = SlidingWindow(df, windowsize, output_col, input_col, rnn_window=5, 
                       forecast=5)
    assert sw.__len__() == 11

    for i in range(sw.__len__()):
        assert sw.__getitem__(i)[0][0].shape == torch.Size([2, 5])
        assert sw.__getitem__(i)[0][1].shape == torch.Size([2, 5])
        assert sw.__getitem__(i)[1].shape == torch.Size([2, 5])

def test_SlidingWindow_len7(df):
    windowsize = 5
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    sw = SlidingWindow(df, windowsize, output_col, input_col, rnn_window=1, 
                       forecast=2, full_ds=False)
    assert sw.__len__() == 7

    for i in range(sw.__len__()):
        assert sw.__getitem__(i)[0][0].shape == torch.Size([2, 5])
        assert sw.__getitem__(i)[0][1].shape == torch.Size([2, 1])
        assert sw.__getitem__(i)[1].shape == torch.Size([2, 2])

def test_SlidingWindow_len8(df):
    windowsize = 5
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    sw = SlidingWindow(df, windowsize, output_col, input_col, rnn_window=5, 
                       forecast=5, full_ds=False)
    assert sw.__len__() == 3

    for i in range(sw.__len__()):
        assert sw.__getitem__(i)[0][0].shape == torch.Size([2, 5])
        assert sw.__getitem__(i)[0][1].shape == torch.Size([2, 5])
        assert sw.__getitem__(i)[1].shape == torch.Size([2, 5])
        
def test_SlidingWindow_output1(df):
    windowsize = 5
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    sw = SlidingWindow(df, windowsize, output_col, input_col)

    index = 0
    expected_output_x = torch.tensor([[1, 2, 3, 4, 5], [0, 1, 0, 1, 0]],
                                     dtype=torch.float)
    expected_output_y = torch.tensor([[15], [25]], dtype=torch.float)

    out_x, out_y = sw.__getitem__(index)
    torch.testing.assert_close(out_x, expected_output_x)
    torch.testing.assert_close(out_y, expected_output_y)

def test_SlidingWindow_output2(df):
    windowsize = 5
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    sw = SlidingWindow(df, windowsize, output_col, input_col, forecast=1,
                       full_ds=True)

    index = 0
    expected_output_x = torch.tensor([[1, 2, 3, 4, 5], [0, 1, 0, 1, 0]],
                                     dtype=torch.float)
    expected_output_y = torch.tensor([[15], [25]], dtype=torch.float)

    out_x, out_y = sw.__getitem__(index)
    torch.testing.assert_close(out_x, expected_output_x)
    torch.testing.assert_close(out_y, expected_output_y)


def test_shuffled_inp(df):
    windowsize = 5
    input_col = ['in_col2', 'in_col1']
    output_col = ['out_col1', 'out_col2']
    sw = SlidingWindow(df, windowsize, output_col, input_col)

    index = 0
    expected_output_x = torch.tensor([[0, 1, 0, 1, 0], [1, 2, 3, 4, 5]],
                                     dtype=torch.float)
    expected_output_y = torch.tensor([[15], [25]], dtype=torch.float)

    out_x, out_y = sw.__getitem__(index)
    torch.testing.assert_close(out_x, expected_output_x)
    torch.testing.assert_close(out_y, expected_output_y)


def test_SlidingWindow_output2(df):
    windowsize = 5
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    sw = SlidingWindow(df, windowsize, output_col, input_col, rnn_window=1)

    index = 0
    expected_output_x1 = torch.tensor([[0, 0, 0, 0, 1], [0, 0, 0, 0, 0]],
                                      dtype=torch.float)
    expected_output_x2 = torch.tensor([[0], [0]], dtype=torch.float)
    expected_output_y = torch.tensor([[11], [21]], dtype=torch.float)

    out_x, out_y = sw.__getitem__(index)
    torch.testing.assert_close(out_x[0], expected_output_x1)
    torch.testing.assert_close(out_x[1], expected_output_x2)
    torch.testing.assert_close(out_y, expected_output_y)

    index = 5
    expected_output_x1 = torch.tensor([[2, 3, 4, 5, 6], [1, 0, 1, 0, 1]],
                                      dtype=torch.float)
    expected_output_x2 = torch.tensor([[15], [25]], dtype=torch.float)
    expected_output_y = torch.tensor([[16], [26]], dtype=torch.float)

    out_x, out_y = sw.__getitem__(index)
    torch.testing.assert_close(out_x[0], expected_output_x1)
    torch.testing.assert_close(out_x[1], expected_output_x2)
    torch.testing.assert_close(out_y, expected_output_y)

    sw = SlidingWindow(df, windowsize, output_col, input_col, rnn_window=3,
                       full_ds=False)
    index = 5
    expected_output_x1 = torch.tensor([[2, 3, 4, 5, 6], [1, 0, 1, 0, 1]],
                                      dtype=torch.float)
    expected_output_x2 = torch.tensor([[13, 14, 15], [23, 24, 25]],
                                      dtype=torch.float)
    expected_output_y = torch.tensor([[16], [26]], dtype=torch.float)

    out_x, out_y = sw.__getitem__(index)

    torch.testing.assert_close(out_x[0], expected_output_x1)
    torch.testing.assert_close(out_x[1], expected_output_x2)
    torch.testing.assert_close(out_y, expected_output_y)

def test_SlidingWindow_output2_precomp(df):
    windowsize = 5
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    sw = SlidingWindow(df, windowsize, output_col, input_col, rnn_window=1, 
                       pre_comp=['out_col1_precomp', 'out_col2_precomp'])

    index = 0
    expected_output_x1 = torch.tensor([[0, 0, 0, 0, 1], [0, 0, 0, 0, 0]],
                                      dtype=torch.float)
    expected_output_x2 = torch.tensor([[0], [0]], dtype=torch.float)
    expected_output_y = torch.tensor([[11], [21]], dtype=torch.float)

    out_x, out_y = sw.__getitem__(index)
    torch.testing.assert_close(out_x[0], expected_output_x1)
    torch.testing.assert_close(out_x[1], expected_output_x2)
    torch.testing.assert_close(out_y, expected_output_y)

    index = 5
    expected_output_x1 = torch.tensor([[2, 3, 4, 5, 6], [1, 0, 1, 0, 1]],
                                      dtype=torch.float)
    expected_output_x2 = torch.tensor([[115], [125]], dtype=torch.float)
    expected_output_y = torch.tensor([[16], [26]], dtype=torch.float)

    out_x, out_y = sw.__getitem__(index)
    print(out_x)
    torch.testing.assert_close(out_x[0], expected_output_x1)
    torch.testing.assert_close(out_x[1], expected_output_x2)
    torch.testing.assert_close(out_y, expected_output_y)

    sw = SlidingWindow(df, windowsize, output_col, input_col, rnn_window=3,
                       full_ds=False, pre_comp=['out_col1_precomp', 'out_col2_precomp'])
    index = 5
    expected_output_x1 = torch.tensor([[2, 3, 4, 5, 6], [1, 0, 1, 0, 1]],
                                      dtype=torch.float)
    expected_output_x2 = torch.tensor([[113, 114, 115], [123, 124, 125]],
                                      dtype=torch.float)
    expected_output_y = torch.tensor([[16], [26]], dtype=torch.float)

    out_x, out_y = sw.__getitem__(index)
    print(out_x)
    torch.testing.assert_close(out_x[0], expected_output_x1)
    torch.testing.assert_close(out_x[1], expected_output_x2)
    torch.testing.assert_close(out_y, expected_output_y)


def test_SlidingWindow_forecast(df):
    windowsize = 5
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    sw = SlidingWindow(df, windowsize, output_col, input_col, rnn_window=1,
                       forecast=2)

    index = 0
    expected_output_x1 = torch.tensor([[0, 0, 0, 1, 2], [0, 0, 0, 0, 1]],
                                      dtype=torch.float)
    expected_output_x2 = torch.tensor([[0], [0]], dtype=torch.float)
    expected_output_y = torch.tensor([[11, 12], [21, 22]], dtype=torch.float)

    out_x, out_y = sw.__getitem__(index)
    torch.testing.assert_close(out_x[0], expected_output_x1)
    torch.testing.assert_close(out_x[1], expected_output_x2)
    torch.testing.assert_close(out_y, expected_output_y)

    index = 5
    expected_output_x1 = torch.tensor([[3, 4, 5, 6, 7], [0, 1, 0, 1, 0]],
                                      dtype=torch.float)
    expected_output_x2 = torch.tensor([[15], [25]], dtype=torch.float)
    expected_output_y = torch.tensor([[16, 17], [26, 27]], dtype=torch.float)

    out_x, out_y = sw.__getitem__(index)
    torch.testing.assert_close(out_x[0], expected_output_x1)
    torch.testing.assert_close(out_x[1], expected_output_x2)
    torch.testing.assert_close(out_y, expected_output_y)


def test_SlidingWindow_forecast2(df):
    windowsize = 5
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    sw = SlidingWindow(df, windowsize, output_col, input_col, rnn_window=1,
                       forecast=2, full_ds=False)

    index = 0
    expected_output_x1 = torch.tensor([[0, 0, 0, 1, 2], [0, 0, 0, 0, 1]],
                                      dtype=torch.float)
    expected_output_x2 = torch.tensor([[0], [0]], dtype=torch.float)
    expected_output_y = torch.tensor([[11, 12], [21, 22]], dtype=torch.float)

    out_x, out_y = sw.__getitem__(index)
    torch.testing.assert_close(out_x[0], expected_output_x1)
    torch.testing.assert_close(out_x[1], expected_output_x2)
    torch.testing.assert_close(out_y, expected_output_y)

    index = 1
    expected_output_x1 = torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 0, 1]],
                                      dtype=torch.float)
    expected_output_x2 = torch.tensor([[12], [22]], dtype=torch.float)
    expected_output_y = torch.tensor([[13, 14], [23, 24]], dtype=torch.float)

    out_x, out_y = sw.__getitem__(index)
    torch.testing.assert_close(out_x[0], expected_output_x1)
    torch.testing.assert_close(out_x[1], expected_output_x2)
    torch.testing.assert_close(out_y, expected_output_y)

    index = sw.__len__() - 1
    expected_output_x1 = torch.tensor([[10, 11, 12, 13, 0], [1, 0, 1, 0, 0]],
                                      dtype=torch.float)
    expected_output_x2 = torch.tensor([[22], [32]], dtype=torch.float)
    expected_output_y = torch.tensor([[23, 0], [33, 0]], dtype=torch.float)

    out_x, out_y = sw.__getitem__(index)
    torch.testing.assert_close(out_x[0], expected_output_x1)
    torch.testing.assert_close(out_x[1], expected_output_x2)
    torch.testing.assert_close(out_y, expected_output_y)


def test_SlidingWindow_forecast2_precomp(df):
    windowsize = 5
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    sw = SlidingWindow(df, windowsize, output_col, input_col, rnn_window=1,
                       forecast=2, full_ds=False, pre_comp=['out_col1_precomp', 'out_col2_precomp'])

    index = 0
    expected_output_x1 = torch.tensor([[0, 0, 0, 1, 2], [0, 0, 0, 0, 1]],
                                      dtype=torch.float)
    expected_output_x2 = torch.tensor([[0], [0]], dtype=torch.float)
    expected_output_y = torch.tensor([[11, 12], [21, 22]], dtype=torch.float)

    out_x, out_y = sw.__getitem__(index)
    torch.testing.assert_close(out_x[0], expected_output_x1)
    torch.testing.assert_close(out_x[1], expected_output_x2)
    torch.testing.assert_close(out_y, expected_output_y)

    index = 1
    expected_output_x1 = torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 0, 1]],
                                      dtype=torch.float)
    expected_output_x2 = torch.tensor([[112], [122]], dtype=torch.float)
    expected_output_y = torch.tensor([[13, 14], [23, 24]], dtype=torch.float)

    out_x, out_y = sw.__getitem__(index)
    torch.testing.assert_close(out_x[0], expected_output_x1)
    torch.testing.assert_close(out_x[1], expected_output_x2)
    torch.testing.assert_close(out_y, expected_output_y)

    index = sw.__len__() - 1
    expected_output_x1 = torch.tensor([[10, 11, 12, 13, 0], [1, 0, 1, 0, 0]],
                                      dtype=torch.float)
    expected_output_x2 = torch.tensor([[122], [132]], dtype=torch.float)
    expected_output_y = torch.tensor([[23, 0], [33, 0]], dtype=torch.float)

    out_x, out_y = sw.__getitem__(index)
    torch.testing.assert_close(out_x[0], expected_output_x1)
    torch.testing.assert_close(out_x[1], expected_output_x2)
    torch.testing.assert_close(out_y, expected_output_y)

'''
Testing of batched sliding window
'''


def df_l(length=13):
    d = {'in_col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
         'in_col2': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
         'out_col1': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
         'out_col2': [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
         }

    df = pd.DataFrame(d)[:length]

    return df


def test_output():
    length = [13, 3, 5]

    list_SW = []
    for le in length:
        data = SlidingWindow(df_l(le), 5, ['out_col1', 'out_col2'],
                             ['in_col1', 'in_col2'],
                             rnn_window=6)
        list_SW.append(data)

    bsw = batch_rec_SW(list_SW)

    assert bsw.__len__() == 13
    assert bsw.__lengths__() == [13, 5, 3]
    assert bsw.__widths__() == [3, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]

    index = 0
    expected_output_x1 = torch.tensor([[[0, 0, 0, 0, 1], [0, 0, 0, 0, 0]],
                                       [[0, 0, 0, 0, 1], [0, 0, 0, 0, 0]],
                                       [[0, 0, 0, 0, 1], [0, 0, 0, 0, 0]]],
                                      dtype=torch.float)

    expected_output_x2 = torch.tensor([[[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                                       [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                                       [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]],
                                      dtype=torch.float)

    expected_output_y = torch.tensor([[[11], [21]],
                                      [[11], [21]],
                                      [[11], [21]]], dtype=torch.float)

    out_x, out_y = bsw.__getitem__(index)
    
    torch.testing.assert_close(out_x[0], expected_output_x1)
    torch.testing.assert_close(out_x[1], expected_output_x2)
    torch.testing.assert_close(out_y, expected_output_y)
    
    index = 3
    expected_output_x1 = torch.tensor([[[0, 1, 2, 3, 4], [0, 0, 1, 0, 1]],
                                       [[0, 1, 2, 3, 4], [0, 0, 1, 0, 1]]],
                                      dtype=torch.float)

    expected_output_x2 = torch.tensor([[[0, 0, 0, 11, 12, 13], [0, 0, 0, 21, 22, 23]],
                                       [[0, 0, 0, 11, 12, 13], [0, 0, 0, 21, 22, 23]]],
                                      dtype=torch.float)

    expected_output_y = torch.tensor([[[14], [24]], 
                                      [[14], [24]]], dtype=torch.float)

    out_x, out_y = bsw.__getitem__(index)

    torch.testing.assert_close(out_x[0], expected_output_x1)
    torch.testing.assert_close(out_x[1], expected_output_x2)
    torch.testing.assert_close(out_y, expected_output_y)

def test_output_forecast():
    length = [13, 3, 5]

    list_SW = []
    for le in length:
        data = SlidingWindow(df_l(le), 5, ['out_col1', 'out_col2'],
                             ['in_col1', 'in_col2'],
                             rnn_window=6, forecast=3, full_ds=False)
        list_SW.append(data)

    bsw = batch_rec_SW(list_SW)

    assert bsw.__len__() == 5
    assert bsw.__lengths__() == [5, 2, 1]
    assert bsw.__widths__() == [3, 2, 1, 1, 1]

    index = 0
    expected_output_x1 = torch.tensor([[[0, 0, 1, 2, 3], [0, 0, 0, 1, 0]],
                                       [[0, 0, 1, 2, 3], [0, 0, 0, 1, 0]],
                                       [[0, 0, 1, 2, 3], [0, 0, 0, 1, 0]]],
                                      dtype=torch.float)

    expected_output_x2 = torch.tensor([[[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                                       [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                                       [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]],
                                      dtype=torch.float)

    expected_output_y = torch.tensor([[[11, 12, 13], [21, 22, 23]],
                                      [[11, 12, 13], [21, 22, 23]],
                                      [[11, 12, 13], [21, 22, 23]]], dtype=torch.float)

    out_x, out_y = bsw.__getitem__(index)
    
    torch.testing.assert_close(out_x[0], expected_output_x1)
    torch.testing.assert_close(out_x[1], expected_output_x2)
    torch.testing.assert_close(out_y, expected_output_y)
    
    index = 1
    expected_output_x1 = torch.tensor([[[2, 3, 4, 5, 6], [1, 0, 1, 0, 1]],
                                       [[2, 3, 4, 5, 0], [1, 0, 1, 0, 0]]],
                                      dtype=torch.float)

    expected_output_x2 = torch.tensor([[[0, 0, 0, 11, 12, 13], [0, 0, 0, 21, 22, 23]],
                                       [[0, 0, 0, 11, 12, 13], [0, 0, 0, 21, 22, 23]]],
                                      dtype=torch.float)

    expected_output_y = torch.tensor([[[14, 15, 16], [24, 25, 26]], 
                                      [[14, 15, 0], [24, 25, 0]]], dtype=torch.float)

    out_x, out_y = bsw.__getitem__(index)

    torch.testing.assert_close(out_x[0], expected_output_x1)
    torch.testing.assert_close(out_x[1], expected_output_x2)
    torch.testing.assert_close(out_y, expected_output_y)