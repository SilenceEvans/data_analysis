{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "2b76c49f",
   "metadata": {},
   "source": [
    "# ndarray数组"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 133,
   "id": "dc766206",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "80ca7239",
   "metadata": {},
   "source": [
    "## 数组加数字、数组加数组"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 134,
   "id": "86089839",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'numpy.ndarray'>\n"
     ]
    }
   ],
   "source": [
    "nd_array = np.array([1,2,3,4,5])\n",
    "print(type(nd_array))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 135,
   "id": "67b0fdc5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([2, 3, 4, 5, 6])"
      ]
     },
     "execution_count": 135,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "nd_array1 = nd_array + 1\n",
    "nd_array1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 136,
   "id": "f50f24a8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 3,  5,  7,  9, 11])"
      ]
     },
     "execution_count": 136,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "nd_array2 = nd_array + nd_array1\n",
    "nd_array2"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "aa0feed4",
   "metadata": {},
   "source": [
    "## 数组特性"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c440ec88",
   "metadata": {},
   "source": [
    "### 看数组的维度 shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 137,
   "id": "a818ea99",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(5,)"
      ]
     },
     "execution_count": 137,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "nd_array.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "796f9b3e",
   "metadata": {},
   "source": [
    "### 数组中数据如果有低级别的数据，则所有数句都转为低级别数据 int>float>str"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 138,
   "id": "43789fe3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['1', '2', '3', '4'], dtype='<U21')"
      ]
     },
     "execution_count": 138,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 全转为str\n",
    "np.array([1,2,3,'4'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 139,
   "id": "6008c4d1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1. , 2. , 3. , 1.5])"
      ]
     },
     "execution_count": 139,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 全转为float\n",
    "np.array([1,2,3,1.5])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "357f6710",
   "metadata": {},
   "source": [
    "## 获取数组属性操作"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 140,
   "id": "5d6c7d2c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "numpy.ndarray"
      ]
     },
     "execution_count": 140,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 获取数组类型，是array还是ndarray?\n",
    "type_array = np.array([1,2,3,4])\n",
    "type(type_array)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 141,
   "id": "6dfa3a43",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dtype('int64')"
      ]
     },
     "execution_count": 141,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 获取数组中数据类型 dtype属性\n",
    "type_array.dtype"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 142,
   "id": "1ed991a1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "4"
      ]
     },
     "execution_count": 142,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 获取当前数组中元素个数\n",
    "type_array.size"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 143,
   "id": "a926be4f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 143,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 获取当前数组的维度\n",
    "type_array.ndim"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b1960d4d",
   "metadata": {},
   "source": [
    "以上获取数组种元素类型，个数，维度的操作应该是属于ndarray调取自己的三个属性"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d22ef58e",
   "metadata": {},
   "source": [
    "# 索引与切片"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a65d4ca7",
   "metadata": {},
   "source": [
    "## 数值索引"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 144,
   "id": "78909ecd",
   "metadata": {},
   "outputs": [],
   "source": [
    "index_array = np.array([1,2,3,4,5])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 145,
   "id": "be144217",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([3, 4])"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 获取索引为2到4位置的元素\n",
    "index_array[2:4]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 146,
   "id": "f4294dfc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([3, 4, 5])"
      ]
     },
     "execution_count": 146,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 获取倒数三个位置的元素\n",
    "index_array[-3:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 147,
   "id": "c5427fe9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 2, 6, 4, 5])"
      ]
     },
     "execution_count": 147,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 将索引为2处的元素赋值为6\n",
    "index_array[2] = 6\n",
    "index_array"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 148,
   "id": "8674fd48",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1,  2,  3],\n",
       "       [ 4, 10,  6],\n",
       "       [ 7,  8,  9]])"
      ]
     },
     "execution_count": 148,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 同理也可以用索引操作处理多维数组\n",
    "dim_array = np.array([[1,2,3]\n",
    "                      ,[4,5,6]\n",
    "                      ,[7,8,9]])\n",
    "# 修改第2行第2列数据为10\n",
    "dim_array[1,1] = 10\n",
    "dim_array"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 149,
   "id": "2ef647fa",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 2, 10,  8])"
      ]
     },
     "execution_count": 149,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 获取第2列的数据\n",
    "dim_array[:,1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 150,
   "id": "c14c9f37",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 4, 10,  6])"
      ]
     },
     "execution_count": 150,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 获取第2行的数据,以下两种方式都可以\n",
    "dim_array[1,:]\n",
    "dim_array[1]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a6e7150a",
   "metadata": {},
   "source": [
    "## 布尔索引"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 151,
   "id": "a74786d1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([2, 5])"
      ]
     },
     "execution_count": 151,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 自己创建布尔索引,输出自定义索引为true未知的元素\n",
    "np_array = np.array([1,2,3,4,5])\n",
    "bool_index = np.array([0,1,0,0,1],dtype = bool)\n",
    "np_array[bool_index]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 152,
   "id": "c561da42",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.22479665 0.19806286 0.76053071 0.16911084 0.08833981 0.68535982\n",
      " 0.95339335 0.00394827 0.51219226 0.81262096]\n",
      "[False False  True False False  True  True False  True  True]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([0.76053071, 0.68535982, 0.95339335, 0.51219226, 0.81262096])"
      ]
     },
     "execution_count": 152,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 随机输出（0，1）之间的10个数\n",
    "rand_array = np.random.rand(10)\n",
    "print(rand_array)\n",
    "# 获取其中大于0.5的数的布尔值\n",
    "mask = rand_array > 0.5\n",
    "print(mask)\n",
    "condition_array = rand_array[mask]\n",
    "condition_array"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 153,
   "id": "f5019d12",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(array([2, 5, 6, 8, 9]),)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([0.76053071, 0.68535982, 0.95339335, 0.51219226, 0.81262096])"
      ]
     },
     "execution_count": 153,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 或者直接将判断条件放置于数组中\n",
    "# 找到符合要求的位置处的索引\n",
    "condition1_array = np.where(rand_array > 0.5)\n",
    "print(condition1_array)\n",
    "\n",
    "#找到符合要求处索引的元素\n",
    "condition2_array = rand_array[condition1_array]\n",
    "condition2_array"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 154,
   "id": "8298f937",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ True,  True,  True, False])"
      ]
     },
     "execution_count": 154,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 实现两个数组的逻辑判断\n",
    "x = np.array([1,1,1,0])\n",
    "y = np.array([1,1,1,2])\n",
    "x == y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 155,
   "id": "9699e976",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ True,  True,  True, False])"
      ]
     },
     "execution_count": 155,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 进行逻辑运算\n",
    "np.logical_and(x,y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 156,
   "id": "1d434851",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ True,  True,  True,  True])"
      ]
     },
     "execution_count": 156,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.logical_or(x,y)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5c791f82",
   "metadata": {},
   "source": [
    "## 数据类型与数值计算"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2c3409e2",
   "metadata": {},
   "source": [
    "### 数据类型"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 157,
   "id": "4a886cd5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1. 2. 3. 4.]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array(['1', '2', '3', '4'], dtype=object)"
      ]
     },
     "execution_count": 157,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#指定元素的数据类型\n",
    "type_array = np.array([1,2,3,4],dtype = float)\n",
    "print(type_array)\n",
    "np.array(['1','2','3','4'],dtype = object)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 158,
   "id": "83791da6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1., 2., 3., 4.])"
      ]
     },
     "execution_count": 158,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 对创建好的数组将进行类型转换\n",
    "array = np.array([1,2,3,4])\n",
    "transfer_array = np.asarray(array,dtype = float)\n",
    "transfer_array"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "33eedb6c",
   "metadata": {},
   "source": [
    "### 复制与赋值"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 159,
   "id": "48648a35",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 2, 3, 4])"
      ]
     },
     "execution_count": 159,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "array = np.array([1,2,3,4])\n",
    "copy_array = array\n",
    "array"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 160,
   "id": "73936bd8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 1 10  3  4]\n",
      "[ 1 10  3  4]\n"
     ]
    }
   ],
   "source": [
    "# 改变其中某一个值得变量观察两个数组\n",
    "copy_array[1] = 10\n",
    "print(array)\n",
    "print(copy_array)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0114eae6",
   "metadata": {},
   "source": [
    "说明这两个变量本质上是一个变量，要想复制后得变量是一个新得变量，不是对之前变量得引用，该怎么办？"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 161,
   "id": "f0a438ca",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 1 10  3  4]\n",
      "[1 2 3 4]\n"
     ]
    }
   ],
   "source": [
    "# 使用copy方法\n",
    "copy1_array = np.copy(array)\n",
    "copy1_array[1] = 2\n",
    "print(array)\n",
    "print(copy1_array)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c4589926",
   "metadata": {},
   "source": [
    "### 数值运算"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 162,
   "id": "32c75b05",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "55"
      ]
     },
     "execution_count": 162,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 1.把数组中得全部元素加起来求和\n",
    "array = np.array([[1,2,3,4,5],[6,7,8,9,10]])\n",
    "sum_array = np.sum(array)\n",
    "sum_array"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 163,
   "id": "66070cc8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 7  9 11 13 15]\n",
      "[15 40]\n"
     ]
    }
   ],
   "source": [
    "# 只对列求和\n",
    "sum_col_sum = np.sum(array,axis = 0)\n",
    "print(sum_col_sum)\n",
    "# 只对行求和\n",
    "sum_row_sum = np.sum(array,axis = 1)\n",
    "print(sum_row_sum)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 164,
   "id": "d245c42d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3628800"
      ]
     },
     "execution_count": 164,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 2.各元素累乘\n",
    "mul_result = array.prod()\n",
    "mul_result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 165,
   "id": "82432bc7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 6 14 24 36 50]\n",
      "[  120 30240]\n"
     ]
    }
   ],
   "source": [
    "# 每列相乘\n",
    "col_mul = array.prod(axis = 0)\n",
    "print(col_mul)\n",
    "# 每行相乘\n",
    "row_mul = array.prod(axis = 1)\n",
    "print(row_mul)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 166,
   "id": "81aeccbf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1, 10)"
      ]
     },
     "execution_count": 166,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 3.求元素中的最值\n",
    "min_value = array.min()\n",
    "max_value = array.max()\n",
    "min_value,max_value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 167,
   "id": "e65875a4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(array([1, 2, 3, 4, 5]), array([1, 6]))"
      ]
     },
     "execution_count": 167,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 求每列、行的最值\n",
    "col_min_value = array.min(axis = 0)\n",
    "row_min_value = array.min(axis = 1)\n",
    "col_min_value,row_min_value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 168,
   "id": "2d5181dc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(array([[ 1,  2,  3,  4,  5],\n",
       "        [ 6,  7,  8,  9, 10]]),\n",
       " 9,\n",
       " 0)"
      ]
     },
     "execution_count": 168,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 求最值所在位置的索引\n",
    "array = np.array([[1,2,3,4,5],[6,7,8,9,10]])\n",
    "index_min = array.argmin()\n",
    "index_max = array.argmax()\n",
    "array,index_max,index_min"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 169,
   "id": "217476c5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5.5"
      ]
     },
     "execution_count": 169,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 4.求均值\n",
    "mean_value = array.mean()\n",
    "mean_value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 170,
   "id": "ed779749",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(array([3.5, 4.5, 5.5, 6.5, 7.5]), array([3., 8.]))"
      ]
     },
     "execution_count": 170,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 求每行每列的均值\n",
    "col_mean = array.mean(axis = 0)\n",
    "row_mean = array.mean(axis = 1)\n",
    "col_mean,row_mean"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 171,
   "id": "4ad4a2b8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2.8722813232690143"
      ]
     },
     "execution_count": 171,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 5.求标准差\n",
    "std_value = array.std()\n",
    "std_value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 172,
   "id": "7b43b7e4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "8.25"
      ]
     },
     "execution_count": 172,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 6.求方差\n",
    "var_value = array.var()\n",
    "var_value"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "237bc455",
   "metadata": {},
   "source": [
    "同样也可以求每行、列的方差、标准差，只需要设置axis就可以"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 173,
   "id": "508d09c9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[5, 5, 5, 5, 5],\n",
       "       [6, 7, 7, 7, 7]])"
      ]
     },
     "execution_count": 173,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 7.设置数组中元素比5小的全部为7，比7大的全部为7\n",
    "array1 = array.clip(5,7)\n",
    "array1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 174,
   "id": "15e46e3c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(array([1., 1., 2., 3., 4.]), array([1. , 1.3, 2.2, 3.2, 3.6]))"
      ]
     },
     "execution_count": 174,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 8.四舍五入\n",
    "array = np.array([1,1.3,2.24,3.25,3.56])\n",
    "array1 = array.round()\n",
    "array_decimals = array.round(decimals=1)\n",
    "array1,array_decimals"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a656c582",
   "metadata": {},
   "source": [
    "### 矩阵乘法"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 175,
   "id": "4102e2be",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([10, 10])"
      ]
     },
     "execution_count": 175,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x = np.array([2,2])\n",
    "y = np.array([5,5])\n",
    "# 对应位置元素相乘\n",
    "np.multiply(x,y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 176,
   "id": "326af65b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "20"
      ]
     },
     "execution_count": 176,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 矩阵乘法\n",
    "np.dot(x,y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 177,
   "id": "0c14621a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[10, 10],\n",
       "       [10, 10]])"
      ]
     },
     "execution_count": 177,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 矩阵乘法，将两个矩阵分别编程可以相乘的维度\n",
    "x.shape = 2,1\n",
    "y.shape = 1,2\n",
    "np.dot(x,y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 178,
   "id": "7f087d2c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[20]])"
      ]
     },
     "execution_count": 178,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.dot(y,x)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "458d701c",
   "metadata": {},
   "source": [
    "## 常用功能模块"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "299df345",
   "metadata": {},
   "source": [
    "### 排序操作"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 179,
   "id": "26b2761a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1.2, 2.7, 3.1, 4.2],\n",
       "       [1.5, 2.2, 3.2, 4.7]])"
      ]
     },
     "execution_count": 179,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sort_array = np.array([[1.2,3.1,4.2,2.7],[2.2,1.5,3.2,4.7]])\n",
    "np.sort(sort_array)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 180,
   "id": "9bcb7c6e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1.2, 1.5, 3.2, 2.7],\n",
       "       [2.2, 3.1, 4.2, 4.7]])"
      ]
     },
     "execution_count": 180,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 也可以按列排序\n",
    "np.sort(sort_array,axis = 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 181,
   "id": "476253e9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 0.          1.11111111  2.22222222  3.33333333  4.44444444  5.55555556\n",
      "  6.66666667  7.77777778  8.88888889 10.        ]\n",
      "[3 6 9]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([ 0.        ,  1.11111111,  2.22222222,  2.5       ,  3.33333333,\n",
       "        4.44444444,  5.55555556,  6.5       ,  6.66666667,  7.77777778,\n",
       "        8.88888889,  9.5       , 10.        ])"
      ]
     },
     "execution_count": 181,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rand_array = np.linspace(0,10,10)\n",
    "print(rand_array)\n",
    "# 往rand_array中插入一组数\n",
    "values = np.array([2.5,6.5,9.5])\n",
    "# 要使得插入后数据的大小顺序符合之前的顺序\n",
    "where_array = np.searchsorted(rand_array,values)\n",
    "print(where_array)\n",
    "np.insert(rand_array,where_array,values,axis = 0)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "789b1958",
   "metadata": {},
   "source": [
    "numpy.insert 函数在给定索引之前，沿给定轴在输入数组中插入值。\n",
    "numpy.insert(arr, obj, values, axis)\n",
    "参数说明：\n",
    "\n",
    "arr：输入数组\n",
    "obj：在其之前插入值的索引\n",
    "values：要插入的值\n",
    "axis：沿着它插入的轴，如果未提供，则输入数组会被展开"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 182,
   "id": "f018f7b8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([2, 3, 0, 1])"
      ]
     },
     "execution_count": 182,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sor1_array = np.array([[1,0,6],[1,7,0],[2,3,1],[2,4,0]])\n",
    "# 按照第一列的数据对每一行进行排列\n",
    "index_sort = np.lexsort([-1 * sor1_array[:,0]])\n",
    "index_sort"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 183,
   "id": "9f720e1a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[2, 3, 1],\n",
       "       [2, 4, 0],\n",
       "       [1, 0, 6],\n",
       "       [1, 7, 0]])"
      ]
     },
     "execution_count": 183,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sor1_array[index_sort]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 184,
   "id": "d1c01780",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 0, 6],\n",
       "       [1, 7, 0],\n",
       "       [2, 3, 1],\n",
       "       [2, 4, 0]])"
      ]
     },
     "execution_count": 184,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 按照第一列元素进行升序排序\n",
    "col_index_sort1 = np.lexsort([sor1_array[:,0]])\n",
    "sor1_array[col_index_sort1]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "814381d1",
   "metadata": {},
   "source": [
    "### 数组形状操作"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 185,
   "id": "b894df33",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])"
      ]
     },
     "execution_count": 185,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "shape_array = np.arange(10)\n",
    "shape_array"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 221,
   "id": "d79d3a22",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(2, 5)"
      ]
     },
     "execution_count": 221,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 将其变成2行5列的矩阵\n",
    "shape_array.shape = 2,5\n",
    "shape_array.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 187,
   "id": "8361e8d5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[0, 1, 2, 3, 4],\n",
       "        [5, 6, 7, 8, 9]]])"
      ]
     },
     "execution_count": 187,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 将其增加一个维度\n",
    "shape_array[np.newaxis,:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 188,
   "id": "7d0f0d1b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0, 1, 2, 3, 4],\n",
       "       [5, 6, 7, 8, 9]])"
      ]
     },
     "execution_count": 188,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 将其去掉一个维度\n",
    "shape_array.squeeze()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 189,
   "id": "761aad6f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0, 5],\n",
       "       [1, 6],\n",
       "       [2, 7],\n",
       "       [3, 8],\n",
       "       [4, 9]])"
      ]
     },
     "execution_count": 189,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#对矩阵进行转置\n",
    "t1_array = shape_array.transpose()\n",
    "t1_array"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 190,
   "id": "17bf2149",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0, 1, 2, 3, 4],\n",
       "       [5, 6, 7, 8, 9]])"
      ]
     },
     "execution_count": 190,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 转置的另一种方法\n",
    "t2_array = t1_array.T\n",
    "t2_array"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7d7baaf1",
   "metadata": {},
   "source": [
    "### 数组的拼接"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 191,
   "id": "e766f02a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1,  2,  3],\n",
       "       [ 4,  5,  6],\n",
       "       [ 7,  8,  9],\n",
       "       [10, 11, 12]])"
      ]
     },
     "execution_count": 191,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a = np.array([[1,2,3],[4,5,6]])\n",
    "b = np.array([[7,8,9],[10,11,12]])\n",
    "# 使用concatenate\n",
    "np.concatenate((a,b))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 192,
   "id": "8f9ec273",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[ 1,  2,  3],\n",
       "        [ 4,  5,  6]],\n",
       "\n",
       "       [[ 7,  8,  9],\n",
       "        [10, 11, 12]]])"
      ]
     },
     "execution_count": 192,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 使用stack，拼接后会增加一个维度\n",
    "np.stack((a,b))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 193,
   "id": "cd511f6d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1,  2,  3,  7,  8,  9],\n",
       "       [ 4,  5,  6, 10, 11, 12]])"
      ]
     },
     "execution_count": 193,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 水平方向拼接 horizotal\n",
    "np.hstack((a,b))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 194,
   "id": "8cb93ea2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1,  2,  3],\n",
       "       [ 4,  5,  6],\n",
       "       [ 7,  8,  9],\n",
       "       [10, 11, 12]])"
      ]
     },
     "execution_count": 194,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 垂直方向拼接 vertical\n",
    "np.vstack((a,b))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 195,
   "id": "9c8dfe9a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 2, 3, 4, 5, 6])"
      ]
     },
     "execution_count": 195,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 降低一个维度，拉平数组\n",
    "a.flatten()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "72c779e9",
   "metadata": {},
   "source": [
    "### 创建数组函数"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 196,
   "id": "3961888c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 2, 3, 4])"
      ]
     },
     "execution_count": 196,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 快速创建行向量\n",
    "np.r_[0:5:1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 197,
   "id": "613b4a9e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0],\n",
       "       [1],\n",
       "       [2],\n",
       "       [3],\n",
       "       [4]])"
      ]
     },
     "execution_count": 197,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 创建列向量\n",
    "np.c_[0:5:1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 198,
   "id": "3f2cf229",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0, 0, 0],\n",
       "       [0, 0, 0],\n",
       "       [0, 0, 0]])"
      ]
     },
     "execution_count": 198,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 创建3*3的0矩阵\n",
    "np.zeros((3,3),dtype = int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 199,
   "id": "50efd565",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 1, 1],\n",
       "       [1, 1, 1],\n",
       "       [1, 1, 1]])"
      ]
     },
     "execution_count": 199,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 创建3*3的单位矩阵\n",
    "np.ones((3,3),dtype = int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 200,
   "id": "a0a58331",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[8, 8, 8],\n",
       "       [8, 8, 8],\n",
       "       [8, 8, 8]])"
      ]
     },
     "execution_count": 200,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 利用单位矩阵生成元素全为8的矩阵\n",
    "np.ones((3,3),dtype = int) * 8"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 201,
   "id": "3d73ce52",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[8, 8, 8],\n",
       "       [8, 8, 8],\n",
       "       [8, 8, 8]])"
      ]
     },
     "execution_count": 201,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 创建空矩阵\n",
    "empty = np.empty((3,3),dtype = int)\n",
    "empty.fill(8)\n",
    "empty"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 202,
   "id": "6f61982c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0, 0, 0],\n",
       "       [0, 0, 0],\n",
       "       [0, 0, 0]])"
      ]
     },
     "execution_count": 202,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 初始化一个矩阵，让其与某个矩阵维度相同\n",
    "np.zeros_like(empty)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 203,
   "id": "8ffa1fdf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 0, 0, 0, 0],\n",
       "       [0, 1, 0, 0, 0],\n",
       "       [0, 0, 1, 0, 0],\n",
       "       [0, 0, 0, 1, 0],\n",
       "       [0, 0, 0, 0, 1]])"
      ]
     },
     "execution_count": 203,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 生成对角矩阵\n",
    "np.identity((5),dtype = int)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "63559dfe",
   "metadata": {},
   "source": [
    "### 随机模块"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 204,
   "id": "5b9f0d02",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.61252607, 0.72175532, 0.29187607, 0.91777412, 0.71457578,\n",
       "       0.54254437, 0.14217005, 0.37334076, 0.67413362, 0.44183317])"
      ]
     },
     "execution_count": 204,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 随机生成（0，1）上10个数\n",
    "np.random.rand(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 205,
   "id": "b75d2cb7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0.43401399, 0.61776698],\n",
       "       [0.51313824, 0.65039718],\n",
       "       [0.60103895, 0.8052232 ]])"
      ]
     },
     "execution_count": 205,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 随机生成（0，1）上维度3*2矩阵\n",
    "np.random.rand(3,2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 206,
   "id": "4901fc41",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[8, 8, 9, 2],\n",
       "       [0, 6, 7, 8],\n",
       "       [1, 7, 1, 4],\n",
       "       [0, 8, 5, 4],\n",
       "       [7, 8, 8, 2]])"
      ]
     },
     "execution_count": 206,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 随机生成（0，10）区间上5 * 4矩阵\n",
    "np.random.randint(10,size = (5,4))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 207,
   "id": "8d48e554",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.2959617068796787"
      ]
     },
     "execution_count": 207,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 只想生成一个随机数值\n",
    "np.random.rand()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 208,
   "id": "b32ec7d2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([2, 8, 8])"
      ]
     },
     "execution_count": 208,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 生成某区间上指定个数的值\n",
    "np.random.randint(0,10,3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 209,
   "id": "cb9c32cb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 8, 7, 4, 0, 3, 2, 5, 9, 6])"
      ]
     },
     "execution_count": 209,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 对某一数据集进行洗牌\n",
    "shuffle_array = np.arange(10)\n",
    "np.random.shuffle(shuffle_array)\n",
    "shuffle_array"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 210,
   "id": "1581e83d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[8 2 5 6 3 1 0 7 4 9]\n",
      "[8 2 5 6 3 1 0 7 4 9]\n"
     ]
    }
   ],
   "source": [
    "# 指定随机种子\n",
    "origin_array1 = np.arange(10)\n",
    "np.random.seed(10)\n",
    "np.random.shuffle(origin_array1)\n",
    "print(origin_array1)\n",
    "\n",
    "# 注意此处得在生成一个和第一次洗牌之前一样的数组\n",
    "origin_array2 = np.arange(10)\n",
    "np.random.seed(10)\n",
    "np.random.shuffle(origin_array2)\n",
    "print(origin_array2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e9a20c82",
   "metadata": {},
   "source": [
    "### 文件读写"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 211,
   "id": "6fe0b337",
   "metadata": {},
   "outputs": [],
   "source": [
    "# nump的savetxt（）\n",
    "array = [1,2,3,4,5]\n",
    "np.savetxt('file.txt',array)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 212,
   "id": "dee2b3f6",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 加载文件\n",
    "data = np.loadtxt('file.txt')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 214,
   "id": "81345c51",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1., 2., 3., 4., 5.])"
      ]
     },
     "execution_count": 214,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 读取文件的时候指定分隔符\n",
    "data1 = np.loadtxt('file.txt',delimiter = ',')\n",
    "data1"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
