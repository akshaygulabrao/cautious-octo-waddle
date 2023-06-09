�!
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%��8"&
exponential_avg_factorfloat%  �?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58��
r
conv2_c5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2_c5/bias
k
!conv2_c5/bias/Read/ReadVariableOpReadVariableOpconv2_c5/bias*
_output_shapes
:*
dtype0
�
conv2_c5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2_c5/kernel
{
#conv2_c5/kernel/Read/ReadVariableOpReadVariableOpconv2_c5/kernel*&
_output_shapes
: *
dtype0
�
bn_c4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_namebn_c4/moving_variance
{
)bn_c4/moving_variance/Read/ReadVariableOpReadVariableOpbn_c4/moving_variance*
_output_shapes
: *
dtype0
z
bn_c4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_namebn_c4/moving_mean
s
%bn_c4/moving_mean/Read/ReadVariableOpReadVariableOpbn_c4/moving_mean*
_output_shapes
: *
dtype0
l

bn_c4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
bn_c4/beta
e
bn_c4/beta/Read/ReadVariableOpReadVariableOp
bn_c4/beta*
_output_shapes
: *
dtype0
n
bn_c4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebn_c4/gamma
g
bn_c4/gamma/Read/ReadVariableOpReadVariableOpbn_c4/gamma*
_output_shapes
: *
dtype0
r
conv2_c4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2_c4/bias
k
!conv2_c4/bias/Read/ReadVariableOpReadVariableOpconv2_c4/bias*
_output_shapes
: *
dtype0
�
conv2_c4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv2_c4/kernel
{
#conv2_c4/kernel/Read/ReadVariableOpReadVariableOpconv2_c4/kernel*&
_output_shapes
:  *
dtype0
�
bn_c3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_namebn_c3/moving_variance
{
)bn_c3/moving_variance/Read/ReadVariableOpReadVariableOpbn_c3/moving_variance*
_output_shapes
: *
dtype0
z
bn_c3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_namebn_c3/moving_mean
s
%bn_c3/moving_mean/Read/ReadVariableOpReadVariableOpbn_c3/moving_mean*
_output_shapes
: *
dtype0
l

bn_c3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
bn_c3/beta
e
bn_c3/beta/Read/ReadVariableOpReadVariableOp
bn_c3/beta*
_output_shapes
: *
dtype0
n
bn_c3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebn_c3/gamma
g
bn_c3/gamma/Read/ReadVariableOpReadVariableOpbn_c3/gamma*
_output_shapes
: *
dtype0
r
conv2_c3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2_c3/bias
k
!conv2_c3/bias/Read/ReadVariableOpReadVariableOpconv2_c3/bias*
_output_shapes
: *
dtype0
�
conv2_c3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2_c3/kernel
{
#conv2_c3/kernel/Read/ReadVariableOpReadVariableOpconv2_c3/kernel*&
_output_shapes
: *
dtype0
�
bn_c2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namebn_c2/moving_variance
{
)bn_c2/moving_variance/Read/ReadVariableOpReadVariableOpbn_c2/moving_variance*
_output_shapes
:*
dtype0
z
bn_c2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namebn_c2/moving_mean
s
%bn_c2/moving_mean/Read/ReadVariableOpReadVariableOpbn_c2/moving_mean*
_output_shapes
:*
dtype0
l

bn_c2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
bn_c2/beta
e
bn_c2/beta/Read/ReadVariableOpReadVariableOp
bn_c2/beta*
_output_shapes
:*
dtype0
n
bn_c2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namebn_c2/gamma
g
bn_c2/gamma/Read/ReadVariableOpReadVariableOpbn_c2/gamma*
_output_shapes
:*
dtype0
r
conv2_c2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2_c2/bias
k
!conv2_c2/bias/Read/ReadVariableOpReadVariableOpconv2_c2/bias*
_output_shapes
:*
dtype0
�
conv2_c2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2_c2/kernel
{
#conv2_c2/kernel/Read/ReadVariableOpReadVariableOpconv2_c2/kernel*&
_output_shapes
:*
dtype0
�
bn_c1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namebn_c1/moving_variance
{
)bn_c1/moving_variance/Read/ReadVariableOpReadVariableOpbn_c1/moving_variance*
_output_shapes
:*
dtype0
z
bn_c1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namebn_c1/moving_mean
s
%bn_c1/moving_mean/Read/ReadVariableOpReadVariableOpbn_c1/moving_mean*
_output_shapes
:*
dtype0
l

bn_c1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
bn_c1/beta
e
bn_c1/beta/Read/ReadVariableOpReadVariableOp
bn_c1/beta*
_output_shapes
:*
dtype0
n
bn_c1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namebn_c1/gamma
g
bn_c1/gamma/Read/ReadVariableOpReadVariableOpbn_c1/gamma*
_output_shapes
:*
dtype0
r
conv2_c1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2_c1/bias
k
!conv2_c1/bias/Read/ReadVariableOpReadVariableOpconv2_c1/bias*
_output_shapes
:*
dtype0
�
conv2_c1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2_c1/kernel
{
#conv2_c1/kernel/Read/ReadVariableOpReadVariableOpconv2_c1/kernel*&
_output_shapes
:*
dtype0
j
	off3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	off3/bias
c
off3/bias/Read/ReadVariableOpReadVariableOp	off3/bias*
_output_shapes
:*
dtype0
z
off3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoff3/kernel
s
off3/kernel/Read/ReadVariableOpReadVariableOpoff3/kernel*&
_output_shapes
:*
dtype0
j
	off2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	off2/bias
c
off2/bias/Read/ReadVariableOpReadVariableOp	off2/bias*
_output_shapes
:*
dtype0
z
off2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameoff2/kernel
s
off2/kernel/Read/ReadVariableOpReadVariableOpoff2/kernel*&
_output_shapes
: *
dtype0
j
	off1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	off1/bias
c
off1/bias/Read/ReadVariableOpReadVariableOp	off1/bias*
_output_shapes
:*
dtype0
z
off1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameoff1/kernel
s
off1/kernel/Read/ReadVariableOpReadVariableOpoff1/kernel*&
_output_shapes
: *
dtype0
j
	cls3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	cls3/bias
c
cls3/bias/Read/ReadVariableOpReadVariableOp	cls3/bias*
_output_shapes
:*
dtype0
z
cls3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namecls3/kernel
s
cls3/kernel/Read/ReadVariableOpReadVariableOpcls3/kernel*&
_output_shapes
:*
dtype0
j
	cls2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	cls2/bias
c
cls2/bias/Read/ReadVariableOpReadVariableOp	cls2/bias*
_output_shapes
:*
dtype0
z
cls2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecls2/kernel
s
cls2/kernel/Read/ReadVariableOpReadVariableOpcls2/kernel*&
_output_shapes
: *
dtype0
j
	cls1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	cls1/bias
c
cls1/bias/Read/ReadVariableOpReadVariableOp	cls1/bias*
_output_shapes
:*
dtype0
z
cls1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecls1/kernel
s
cls1/kernel/Read/ReadVariableOpReadVariableOpcls1/kernel*&
_output_shapes
: *
dtype0
�
serving_default_input_1Placeholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2_c1/kernelconv2_c1/biasbn_c1/gamma
bn_c1/betabn_c1/moving_meanbn_c1/moving_varianceconv2_c2/kernelconv2_c2/biasbn_c2/gamma
bn_c2/betabn_c2/moving_meanbn_c2/moving_varianceconv2_c3/kernelconv2_c3/biasbn_c3/gamma
bn_c3/betabn_c3/moving_meanbn_c3/moving_varianceconv2_c4/kernelconv2_c4/biasbn_c4/gamma
bn_c4/betabn_c4/moving_meanbn_c4/moving_varianceconv2_c5/kernelconv2_c5/biasoff3/kernel	off3/biasoff2/kernel	off2/biasoff1/kernel	off1/biascls3/kernel	cls3/biascls2/kernel	cls2/biascls1/kernel	cls1/bias*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *D
_output_shapes2
0:����������*:����������**H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference_signature_wrapper_3160

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
�
layer-0
 layer_with_weights-0
 layer-1
!layer_with_weights-1
!layer-2
"layer-3
#layer_with_weights-2
#layer-4
$layer_with_weights-3
$layer-5
%layer-6
&layer_with_weights-4
&layer-7
'layer_with_weights-5
'layer-8
(layer-9
)layer_with_weights-6
)layer-10
*layer_with_weights-7
*layer-11
+layer-12
,layer_with_weights-8
,layer-13
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses*
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

9kernel
:bias
 ;_jit_compiled_convolution_op*
�
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses

Bkernel
Cbias
 D_jit_compiled_convolution_op*
�
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses

Kkernel
Lbias
 M_jit_compiled_convolution_op*
�
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses

Tkernel
Ubias
 V_jit_compiled_convolution_op*
�
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses

]kernel
^bias
 __jit_compiled_convolution_op*
�
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses

fkernel
gbias
 h_jit_compiled_convolution_op*
�
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses* 
�
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses* 
�
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses* 
�
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
926
:27
B28
C29
K30
L31
T32
U33
]34
^35
f36
g37*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
918
:19
B20
C21
K22
L23
T24
U25
]26
^27
f28
g29*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 

�serving_default* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 

90
:1*

90
:1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
[U
VARIABLE_VALUEcls1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUE	cls1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

B0
C1*

B0
C1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
[U
VARIABLE_VALUEcls2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUE	cls2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

K0
L1*

K0
L1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
[U
VARIABLE_VALUEcls3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUE	cls3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

T0
U1*

T0
U1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
[U
VARIABLE_VALUEoff1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUE	off1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

]0
^1*

]0
^1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
[U
VARIABLE_VALUEoff2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUE	off2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

f0
g1*

f0
g1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
[U
VARIABLE_VALUEoff3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUE	off3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
OI
VARIABLE_VALUEconv2_c1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2_c1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEbn_c1/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
bn_c1/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEbn_c1/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEbn_c1/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2_c2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2_c2/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEbn_c2/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
bn_c2/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEbn_c2/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEbn_c2/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2_c3/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2_c3/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEbn_c3/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
bn_c3/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEbn_c3/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEbn_c3/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2_c4/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2_c4/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEbn_c4/gamma'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
bn_c4/beta'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEbn_c4/moving_mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEbn_c4/moving_variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2_c5/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2_c5/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
D
�0
�1
�2
�3
�4
�5
�6
�7*
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
D
�0
�1
�2
�3
�4
�5
�6
�7*
j
0
 1
!2
"3
#4
$5
%6
&7
'8
(9
)10
*11
+12
,13*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamecls1/kernel/Read/ReadVariableOpcls1/bias/Read/ReadVariableOpcls2/kernel/Read/ReadVariableOpcls2/bias/Read/ReadVariableOpcls3/kernel/Read/ReadVariableOpcls3/bias/Read/ReadVariableOpoff1/kernel/Read/ReadVariableOpoff1/bias/Read/ReadVariableOpoff2/kernel/Read/ReadVariableOpoff2/bias/Read/ReadVariableOpoff3/kernel/Read/ReadVariableOpoff3/bias/Read/ReadVariableOp#conv2_c1/kernel/Read/ReadVariableOp!conv2_c1/bias/Read/ReadVariableOpbn_c1/gamma/Read/ReadVariableOpbn_c1/beta/Read/ReadVariableOp%bn_c1/moving_mean/Read/ReadVariableOp)bn_c1/moving_variance/Read/ReadVariableOp#conv2_c2/kernel/Read/ReadVariableOp!conv2_c2/bias/Read/ReadVariableOpbn_c2/gamma/Read/ReadVariableOpbn_c2/beta/Read/ReadVariableOp%bn_c2/moving_mean/Read/ReadVariableOp)bn_c2/moving_variance/Read/ReadVariableOp#conv2_c3/kernel/Read/ReadVariableOp!conv2_c3/bias/Read/ReadVariableOpbn_c3/gamma/Read/ReadVariableOpbn_c3/beta/Read/ReadVariableOp%bn_c3/moving_mean/Read/ReadVariableOp)bn_c3/moving_variance/Read/ReadVariableOp#conv2_c4/kernel/Read/ReadVariableOp!conv2_c4/bias/Read/ReadVariableOpbn_c4/gamma/Read/ReadVariableOpbn_c4/beta/Read/ReadVariableOp%bn_c4/moving_mean/Read/ReadVariableOp)bn_c4/moving_variance/Read/ReadVariableOp#conv2_c5/kernel/Read/ReadVariableOp!conv2_c5/bias/Read/ReadVariableOpConst*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *&
f!R
__inference__traced_save_4993
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamecls1/kernel	cls1/biascls2/kernel	cls2/biascls3/kernel	cls3/biasoff1/kernel	off1/biasoff2/kernel	off2/biasoff3/kernel	off3/biasconv2_c1/kernelconv2_c1/biasbn_c1/gamma
bn_c1/betabn_c1/moving_meanbn_c1/moving_varianceconv2_c2/kernelconv2_c2/biasbn_c2/gamma
bn_c2/betabn_c2/moving_meanbn_c2/moving_varianceconv2_c3/kernelconv2_c3/biasbn_c3/gamma
bn_c3/betabn_c3/moving_meanbn_c3/moving_varianceconv2_c4/kernelconv2_c4/biasbn_c4/gamma
bn_c4/betabn_c4/moving_meanbn_c4/moving_varianceconv2_c5/kernelconv2_c5/bias*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__traced_restore_5117��
�
�
B__inference_conv2_c1_layer_call_and_return_conditional_losses_4479

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:q
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:�����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
>__inference_off1_layer_call_and_return_conditional_losses_2062

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:o
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������  g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������  w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
l
B__inference_off_cat2_layer_call_and_return_conditional_losses_2228

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :z
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*,
_output_shapes
:����������\
IdentityIdentityconcat:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs
�

^
B__inference_cls_res1_layer_call_and_return_conditional_losses_2210

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:i
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:���������� ]
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:���������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������  :W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
$__inference_bn_c1_layer_call_fn_4505

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_bn_c1_layer_call_and_return_conditional_losses_1047�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
^
B__inference_cls_out2_layer_call_and_return_conditional_losses_2251

inputs
identityQ
SoftmaxSoftmaxinputs*
T0*,
_output_shapes
:����������^
IdentityIdentitySoftmax:softmax:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_conv2_c5_layer_call_fn_4842

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2_c5_layer_call_and_return_conditional_losses_1435w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4645

inputs
identity�
MaxPoolMaxPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�	
'__inference_ssd_head_layer_call_fn_2363
input_1!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17:  

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23: 

unknown_24:$

unknown_25:

unknown_26:$

unknown_27: 

unknown_28:$

unknown_29: 

unknown_30:$

unknown_31:

unknown_32:$

unknown_33: 

unknown_34:$

unknown_35: 

unknown_36:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *D
_output_shapes2
0:����������*:����������**H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_ssd_head_layer_call_and_return_conditional_losses_2282t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������*v

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*,
_output_shapes
:����������*`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�

^
B__inference_cls_res3_layer_call_and_return_conditional_losses_2180

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:i
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:����������]
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1219

inputs
identity�
MaxPoolMaxPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
C
'__inference_cls_res1_layer_call_fn_4255

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_cls_res1_layer_call_and_return_conditional_losses_2210e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:���������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������  :W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�	
'__inference_ssd_head_layer_call_fn_2867
input_1!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17:  

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23: 

unknown_24:$

unknown_25:

unknown_26:$

unknown_27: 

unknown_28:$

unknown_29: 

unknown_30:$

unknown_31:

unknown_32:$

unknown_33: 

unknown_34:$

unknown_35: 

unknown_36:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *D
_output_shapes2
0:����������*:����������**@
_read_only_resource_inputs"
 	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_ssd_head_layer_call_and_return_conditional_losses_2703t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������*v

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*,
_output_shapes
:����������*`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
�
?__inference_bn_c3_layer_call_and_return_conditional_losses_4711

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�

^
B__inference_cls_res3_layer_call_and_return_conditional_losses_4304

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:i
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:����������]
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
>__inference_cls1_layer_call_and_return_conditional_losses_4145

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:o
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������  g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������  w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
�
>__inference_off3_layer_call_and_return_conditional_losses_4250

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:o
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
^
B__inference_cls_out1_layer_call_and_return_conditional_losses_2244

inputs
identityQ
SoftmaxSoftmaxinputs*
T0*,
_output_shapes
:���������� ^
IdentityIdentitySoftmax:softmax:0*
T0*,
_output_shapes
:���������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�@
�	
?__inference_model_layer_call_and_return_conditional_losses_1952
	input_map'
conv2_c1_1884:
conv2_c1_1886:

bn_c1_1889:

bn_c1_1891:

bn_c1_1893:

bn_c1_1895:'
conv2_c2_1899:
conv2_c2_1901:

bn_c2_1904:

bn_c2_1906:

bn_c2_1908:

bn_c2_1910:'
conv2_c3_1914: 
conv2_c3_1916: 

bn_c3_1919: 

bn_c3_1921: 

bn_c3_1923: 

bn_c3_1925: '
conv2_c4_1929:  
conv2_c4_1931: 

bn_c4_1934: 

bn_c4_1936: 

bn_c4_1938: 

bn_c4_1940: '
conv2_c5_1944: 
conv2_c5_1946:
identity

identity_1

identity_2��bn_c1/StatefulPartitionedCall�bn_c2/StatefulPartitionedCall�bn_c3/StatefulPartitionedCall�bn_c4/StatefulPartitionedCall� conv2_c1/StatefulPartitionedCall� conv2_c2/StatefulPartitionedCall� conv2_c3/StatefulPartitionedCall� conv2_c4/StatefulPartitionedCall� conv2_c5/StatefulPartitionedCallk
conv2_c1/CastCast	input_map*

DstT0*

SrcT0*1
_output_shapes
:������������
 conv2_c1/StatefulPartitionedCallStatefulPartitionedCallconv2_c1/Cast:y:0conv2_c1_1884conv2_c1_1886*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2_c1_layer_call_and_return_conditional_losses_1319�
bn_c1/StatefulPartitionedCallStatefulPartitionedCall)conv2_c1/StatefulPartitionedCall:output:0
bn_c1_1889
bn_c1_1891
bn_c1_1893
bn_c1_1895*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_bn_c1_layer_call_and_return_conditional_losses_1047�
max_pooling2d/PartitionedCallPartitionedCall&bn_c1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1067�
 conv2_c2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2_c2_1899conv2_c2_1901*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2_c2_layer_call_and_return_conditional_losses_1348�
bn_c2/StatefulPartitionedCallStatefulPartitionedCall)conv2_c2/StatefulPartitionedCall:output:0
bn_c2_1904
bn_c2_1906
bn_c2_1908
bn_c2_1910*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_bn_c2_layer_call_and_return_conditional_losses_1123�
max_pooling2d_1/PartitionedCallPartitionedCall&bn_c2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1143�
 conv2_c3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2_c3_1914conv2_c3_1916*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2_c3_layer_call_and_return_conditional_losses_1377�
bn_c3/StatefulPartitionedCallStatefulPartitionedCall)conv2_c3/StatefulPartitionedCall:output:0
bn_c3_1919
bn_c3_1921
bn_c3_1923
bn_c3_1925*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_bn_c3_layer_call_and_return_conditional_losses_1199�
max_pooling2d_2/PartitionedCallPartitionedCall&bn_c3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1219�
 conv2_c4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2_c4_1929conv2_c4_1931*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2_c4_layer_call_and_return_conditional_losses_1406�
bn_c4/StatefulPartitionedCallStatefulPartitionedCall)conv2_c4/StatefulPartitionedCall:output:0
bn_c4_1934
bn_c4_1936
bn_c4_1938
bn_c4_1940*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_bn_c4_layer_call_and_return_conditional_losses_1275�
max_pooling2d_3/PartitionedCallPartitionedCall&bn_c4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1295�
 conv2_c5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2_c5_1944conv2_c5_1946*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2_c5_layer_call_and_return_conditional_losses_1435}
IdentityIdentity&bn_c3/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������   

Identity_1Identity&bn_c4/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� �

Identity_2Identity)conv2_c5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^bn_c1/StatefulPartitionedCall^bn_c2/StatefulPartitionedCall^bn_c3/StatefulPartitionedCall^bn_c4/StatefulPartitionedCall!^conv2_c1/StatefulPartitionedCall!^conv2_c2/StatefulPartitionedCall!^conv2_c3/StatefulPartitionedCall!^conv2_c4/StatefulPartitionedCall!^conv2_c5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : 2>
bn_c1/StatefulPartitionedCallbn_c1/StatefulPartitionedCall2>
bn_c2/StatefulPartitionedCallbn_c2/StatefulPartitionedCall2>
bn_c3/StatefulPartitionedCallbn_c3/StatefulPartitionedCall2>
bn_c4/StatefulPartitionedCallbn_c4/StatefulPartitionedCall2D
 conv2_c1/StatefulPartitionedCall conv2_c1/StatefulPartitionedCall2D
 conv2_c2/StatefulPartitionedCall conv2_c2/StatefulPartitionedCall2D
 conv2_c3/StatefulPartitionedCall conv2_c3/StatefulPartitionedCall2D
 conv2_c4/StatefulPartitionedCall conv2_c4/StatefulPartitionedCall2D
 conv2_c5/StatefulPartitionedCall conv2_c5/StatefulPartitionedCall:\ X
1
_output_shapes
:�����������
#
_user_specified_name	input_map
�
�
'__inference_conv2_c2_layer_call_fn_4560

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2_c2_layer_call_and_return_conditional_losses_1348w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
l
B__inference_off_cat3_layer_call_and_return_conditional_losses_2237

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :z
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*,
_output_shapes
:����������\
IdentityIdentityconcat:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
$__inference_bn_c4_layer_call_fn_4787

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_bn_c4_layer_call_and_return_conditional_losses_1275�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
$__inference_model_layer_call_fn_3839

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17:  

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23: 

unknown_24:
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:���������   :��������� :���������*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_1444w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������   y

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:��������� y

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�

^
B__inference_off_res2_layer_call_and_return_conditional_losses_2150

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:i
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:����������]
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�	
'__inference_ssd_head_layer_call_fn_3243

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17:  

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23: 

unknown_24:$

unknown_25:

unknown_26:$

unknown_27: 

unknown_28:$

unknown_29: 

unknown_30:$

unknown_31:

unknown_32:$

unknown_33: 

unknown_34:$

unknown_35: 

unknown_36:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *D
_output_shapes2
0:����������*:����������**H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_ssd_head_layer_call_and_return_conditional_losses_2282t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������*v

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*,
_output_shapes
:����������*`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4551

inputs
identity�
MaxPoolMaxPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
?__inference_bn_c3_layer_call_and_return_conditional_losses_1168

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
>__inference_cls1_layer_call_and_return_conditional_losses_2116

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:o
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������  g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������  w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
J
.__inference_max_pooling2d_2_layer_call_fn_4734

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1219�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
C
'__inference_off_res1_layer_call_fn_4309

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_off_res1_layer_call_and_return_conditional_losses_2165e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:���������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������  :W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
?__inference_bn_c4_layer_call_and_return_conditional_losses_4823

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�

^
B__inference_cls_res1_layer_call_and_return_conditional_losses_4268

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:i
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:���������� ]
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:���������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������  :W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
B__inference_conv2_c4_layer_call_and_return_conditional_losses_1406

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:  �
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: o
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:��������� X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:��������� i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
#__inference_cls3_layer_call_fn_4175

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_cls3_layer_call_and_return_conditional_losses_2080w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
J
.__inference_max_pooling2d_1_layer_call_fn_4640

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1143�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
>__inference_off2_layer_call_and_return_conditional_losses_2044

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:o
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�U
�
B__inference_ssd_head_layer_call_and_return_conditional_losses_3075
input_1$

model_2974:

model_2976:

model_2978:

model_2980:

model_2982:

model_2984:$

model_2986:

model_2988:

model_2990:

model_2992:

model_2994:

model_2996:$

model_2998: 

model_3000: 

model_3002: 

model_3004: 

model_3006: 

model_3008: $

model_3010:  

model_3012: 

model_3014: 

model_3016: 

model_3018: 

model_3020: $

model_3022: 

model_3024:#
	off3_3029:
	off3_3031:#
	off2_3034: 
	off2_3036:#
	off1_3039: 
	off1_3041:#
	cls3_3044:
	cls3_3046:#
	cls2_3049: 
	cls2_3051:#
	cls1_3054: 
	cls1_3056:
identity

identity_1��cls1/StatefulPartitionedCall�cls2/StatefulPartitionedCall�cls3/StatefulPartitionedCall�model/StatefulPartitionedCall�off1/StatefulPartitionedCall�off2/StatefulPartitionedCall�off3/StatefulPartitionedCall�
model/StatefulPartitionedCallStatefulPartitionedCallinput_1
model_2974
model_2976
model_2978
model_2980
model_2982
model_2984
model_2986
model_2988
model_2990
model_2992
model_2994
model_2996
model_2998
model_3000
model_3002
model_3004
model_3006
model_3008
model_3010
model_3012
model_3014
model_3016
model_3018
model_3020
model_3022
model_3024*&
Tin
2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:���������   :��������� :���������*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_1688�
off3/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:2	off3_3029	off3_3031*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_off3_layer_call_and_return_conditional_losses_2026�
off2/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:1	off2_3034	off2_3036*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_off2_layer_call_and_return_conditional_losses_2044�
off1/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0	off1_3039	off1_3041*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_off1_layer_call_and_return_conditional_losses_2062�
cls3/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:2	cls3_3044	cls3_3046*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_cls3_layer_call_and_return_conditional_losses_2080�
cls2/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:1	cls2_3049	cls2_3051*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_cls2_layer_call_and_return_conditional_losses_2098�
cls1/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0	cls1_3054	cls1_3056*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_cls1_layer_call_and_return_conditional_losses_2116�
off_res3/PartitionedCallPartitionedCall%off3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_off_res3_layer_call_and_return_conditional_losses_2135�
off_res2/PartitionedCallPartitionedCall%off2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_off_res2_layer_call_and_return_conditional_losses_2150�
off_res1/PartitionedCallPartitionedCall%off1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_off_res1_layer_call_and_return_conditional_losses_2165�
cls_res3/PartitionedCallPartitionedCall%cls3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_cls_res3_layer_call_and_return_conditional_losses_2180�
cls_res2/PartitionedCallPartitionedCall%cls2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_cls_res2_layer_call_and_return_conditional_losses_2195�
cls_res1/PartitionedCallPartitionedCall%cls1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_cls_res1_layer_call_and_return_conditional_losses_2210�
off_cat1/PartitionedCallPartitionedCall!off_res1/PartitionedCall:output:0!off_res1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_off_cat1_layer_call_and_return_conditional_losses_2219�
off_cat2/PartitionedCallPartitionedCall!off_res2/PartitionedCall:output:0!off_res2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_off_cat2_layer_call_and_return_conditional_losses_2228�
off_cat3/PartitionedCallPartitionedCall!off_res3/PartitionedCall:output:0!off_res3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_off_cat3_layer_call_and_return_conditional_losses_2237�
cls_out1/PartitionedCallPartitionedCall!cls_res1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_cls_out1_layer_call_and_return_conditional_losses_2244�
cls_out2/PartitionedCallPartitionedCall!cls_res2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_cls_out2_layer_call_and_return_conditional_losses_2251�
cls_out3/PartitionedCallPartitionedCall!cls_res3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_cls_out3_layer_call_and_return_conditional_losses_2258�
offsets/PartitionedCallPartitionedCall!off_cat1/PartitionedCall:output:0!off_cat2/PartitionedCall:output:0!off_cat3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_offsets_layer_call_and_return_conditional_losses_2268�
classes/PartitionedCallPartitionedCall!cls_out1/PartitionedCall:output:0!cls_out2/PartitionedCall:output:0!cls_out3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_classes_layer_call_and_return_conditional_losses_2278t
IdentityIdentity classes/PartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������*v

Identity_1Identity offsets/PartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������*�
NoOpNoOp^cls1/StatefulPartitionedCall^cls2/StatefulPartitionedCall^cls3/StatefulPartitionedCall^model/StatefulPartitionedCall^off1/StatefulPartitionedCall^off2/StatefulPartitionedCall^off3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
cls1/StatefulPartitionedCallcls1/StatefulPartitionedCall2<
cls2/StatefulPartitionedCallcls2/StatefulPartitionedCall2<
cls3/StatefulPartitionedCallcls3/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2<
off1/StatefulPartitionedCalloff1/StatefulPartitionedCall2<
off2/StatefulPartitionedCalloff2/StatefulPartitionedCall2<
off3/StatefulPartitionedCalloff3/StatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
�
B__inference_conv2_c3_layer_call_and_return_conditional_losses_4667

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: o
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������   X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������   i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
{
A__inference_offsets_layer_call_and_return_conditional_losses_4457
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*,
_output_shapes
:����������*\
IdentityIdentityconcat:output:0*
T0*,
_output_shapes
:����������*"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������� :����������:����������:V R
,
_output_shapes
:���������� 
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs_1:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs_2
�
�
>__inference_off2_layer_call_and_return_conditional_losses_4229

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:o
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
?__inference_bn_c3_layer_call_and_return_conditional_losses_1199

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
S
'__inference_off_cat1_layer_call_fn_4394
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_off_cat1_layer_call_and_return_conditional_losses_2219e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:���������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:���������� :���������� :V R
,
_output_shapes
:���������� 
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:���������� 
"
_user_specified_name
inputs_1
�@
�	
?__inference_model_layer_call_and_return_conditional_losses_1880
	input_map'
conv2_c1_1812:
conv2_c1_1814:

bn_c1_1817:

bn_c1_1819:

bn_c1_1821:

bn_c1_1823:'
conv2_c2_1827:
conv2_c2_1829:

bn_c2_1832:

bn_c2_1834:

bn_c2_1836:

bn_c2_1838:'
conv2_c3_1842: 
conv2_c3_1844: 

bn_c3_1847: 

bn_c3_1849: 

bn_c3_1851: 

bn_c3_1853: '
conv2_c4_1857:  
conv2_c4_1859: 

bn_c4_1862: 

bn_c4_1864: 

bn_c4_1866: 

bn_c4_1868: '
conv2_c5_1872: 
conv2_c5_1874:
identity

identity_1

identity_2��bn_c1/StatefulPartitionedCall�bn_c2/StatefulPartitionedCall�bn_c3/StatefulPartitionedCall�bn_c4/StatefulPartitionedCall� conv2_c1/StatefulPartitionedCall� conv2_c2/StatefulPartitionedCall� conv2_c3/StatefulPartitionedCall� conv2_c4/StatefulPartitionedCall� conv2_c5/StatefulPartitionedCallk
conv2_c1/CastCast	input_map*

DstT0*

SrcT0*1
_output_shapes
:������������
 conv2_c1/StatefulPartitionedCallStatefulPartitionedCallconv2_c1/Cast:y:0conv2_c1_1812conv2_c1_1814*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2_c1_layer_call_and_return_conditional_losses_1319�
bn_c1/StatefulPartitionedCallStatefulPartitionedCall)conv2_c1/StatefulPartitionedCall:output:0
bn_c1_1817
bn_c1_1819
bn_c1_1821
bn_c1_1823*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_bn_c1_layer_call_and_return_conditional_losses_1016�
max_pooling2d/PartitionedCallPartitionedCall&bn_c1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1067�
 conv2_c2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2_c2_1827conv2_c2_1829*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2_c2_layer_call_and_return_conditional_losses_1348�
bn_c2/StatefulPartitionedCallStatefulPartitionedCall)conv2_c2/StatefulPartitionedCall:output:0
bn_c2_1832
bn_c2_1834
bn_c2_1836
bn_c2_1838*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_bn_c2_layer_call_and_return_conditional_losses_1092�
max_pooling2d_1/PartitionedCallPartitionedCall&bn_c2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1143�
 conv2_c3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2_c3_1842conv2_c3_1844*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2_c3_layer_call_and_return_conditional_losses_1377�
bn_c3/StatefulPartitionedCallStatefulPartitionedCall)conv2_c3/StatefulPartitionedCall:output:0
bn_c3_1847
bn_c3_1849
bn_c3_1851
bn_c3_1853*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_bn_c3_layer_call_and_return_conditional_losses_1168�
max_pooling2d_2/PartitionedCallPartitionedCall&bn_c3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1219�
 conv2_c4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2_c4_1857conv2_c4_1859*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2_c4_layer_call_and_return_conditional_losses_1406�
bn_c4/StatefulPartitionedCallStatefulPartitionedCall)conv2_c4/StatefulPartitionedCall:output:0
bn_c4_1862
bn_c4_1864
bn_c4_1866
bn_c4_1868*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_bn_c4_layer_call_and_return_conditional_losses_1244�
max_pooling2d_3/PartitionedCallPartitionedCall&bn_c4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1295�
 conv2_c5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2_c5_1872conv2_c5_1874*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2_c5_layer_call_and_return_conditional_losses_1435}
IdentityIdentity&bn_c3/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������   

Identity_1Identity&bn_c4/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� �

Identity_2Identity)conv2_c5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^bn_c1/StatefulPartitionedCall^bn_c2/StatefulPartitionedCall^bn_c3/StatefulPartitionedCall^bn_c4/StatefulPartitionedCall!^conv2_c1/StatefulPartitionedCall!^conv2_c2/StatefulPartitionedCall!^conv2_c3/StatefulPartitionedCall!^conv2_c4/StatefulPartitionedCall!^conv2_c5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : 2>
bn_c1/StatefulPartitionedCallbn_c1/StatefulPartitionedCall2>
bn_c2/StatefulPartitionedCallbn_c2/StatefulPartitionedCall2>
bn_c3/StatefulPartitionedCallbn_c3/StatefulPartitionedCall2>
bn_c4/StatefulPartitionedCallbn_c4/StatefulPartitionedCall2D
 conv2_c1/StatefulPartitionedCall conv2_c1/StatefulPartitionedCall2D
 conv2_c2/StatefulPartitionedCall conv2_c2/StatefulPartitionedCall2D
 conv2_c3/StatefulPartitionedCall conv2_c3/StatefulPartitionedCall2D
 conv2_c4/StatefulPartitionedCall conv2_c4/StatefulPartitionedCall2D
 conv2_c5/StatefulPartitionedCall conv2_c5/StatefulPartitionedCall:\ X
1
_output_shapes
:�����������
#
_user_specified_name	input_map
�
�
'__inference_conv2_c4_layer_call_fn_4748

inputs!
unknown:  
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2_c4_layer_call_and_return_conditional_losses_1406w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
H
,__inference_max_pooling2d_layer_call_fn_4546

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1067�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
^
B__inference_cls_out3_layer_call_and_return_conditional_losses_2258

inputs
identityQ
SoftmaxSoftmaxinputs*
T0*,
_output_shapes
:����������^
IdentityIdentitySoftmax:softmax:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
C
'__inference_cls_out1_layer_call_fn_4363

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_cls_out1_layer_call_and_return_conditional_losses_2244e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:���������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�

^
B__inference_off_res2_layer_call_and_return_conditional_losses_4340

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:i
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:����������]
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
 __inference__traced_restore_5117
file_prefix6
assignvariableop_cls1_kernel: *
assignvariableop_1_cls1_bias:8
assignvariableop_2_cls2_kernel: *
assignvariableop_3_cls2_bias:8
assignvariableop_4_cls3_kernel:*
assignvariableop_5_cls3_bias:8
assignvariableop_6_off1_kernel: *
assignvariableop_7_off1_bias:8
assignvariableop_8_off2_kernel: *
assignvariableop_9_off2_bias:9
assignvariableop_10_off3_kernel:+
assignvariableop_11_off3_bias:=
#assignvariableop_12_conv2_c1_kernel:/
!assignvariableop_13_conv2_c1_bias:-
assignvariableop_14_bn_c1_gamma:,
assignvariableop_15_bn_c1_beta:3
%assignvariableop_16_bn_c1_moving_mean:7
)assignvariableop_17_bn_c1_moving_variance:=
#assignvariableop_18_conv2_c2_kernel:/
!assignvariableop_19_conv2_c2_bias:-
assignvariableop_20_bn_c2_gamma:,
assignvariableop_21_bn_c2_beta:3
%assignvariableop_22_bn_c2_moving_mean:7
)assignvariableop_23_bn_c2_moving_variance:=
#assignvariableop_24_conv2_c3_kernel: /
!assignvariableop_25_conv2_c3_bias: -
assignvariableop_26_bn_c3_gamma: ,
assignvariableop_27_bn_c3_beta: 3
%assignvariableop_28_bn_c3_moving_mean: 7
)assignvariableop_29_bn_c3_moving_variance: =
#assignvariableop_30_conv2_c4_kernel:  /
!assignvariableop_31_conv2_c4_bias: -
assignvariableop_32_bn_c4_gamma: ,
assignvariableop_33_bn_c4_beta: 3
%assignvariableop_34_bn_c4_moving_mean: 7
)assignvariableop_35_bn_c4_moving_variance: =
#assignvariableop_36_conv2_c5_kernel: /
!assignvariableop_37_conv2_c5_bias:
identity_39��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*�
value�B�'B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*a
valueXBV'B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::*5
dtypes+
)2'[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_cls1_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_cls1_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_cls2_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_cls2_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_cls3_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_cls3_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_off1_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_off1_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_off2_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_off2_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_off3_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_off3_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2_c1_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2_c1_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_bn_c1_gammaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_bn_c1_betaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp%assignvariableop_16_bn_c1_moving_meanIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp)assignvariableop_17_bn_c1_moving_varianceIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv2_c2_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv2_c2_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_bn_c2_gammaIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_bn_c2_betaIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp%assignvariableop_22_bn_c2_moving_meanIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp)assignvariableop_23_bn_c2_moving_varianceIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp#assignvariableop_24_conv2_c3_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp!assignvariableop_25_conv2_c3_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_bn_c3_gammaIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpassignvariableop_27_bn_c3_betaIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp%assignvariableop_28_bn_c3_moving_meanIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp)assignvariableop_29_bn_c3_moving_varianceIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp#assignvariableop_30_conv2_c4_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp!assignvariableop_31_conv2_c4_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOpassignvariableop_32_bn_c4_gammaIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOpassignvariableop_33_bn_c4_betaIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp%assignvariableop_34_bn_c4_moving_meanIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp)assignvariableop_35_bn_c4_moving_varianceIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp#assignvariableop_36_conv2_c5_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp!assignvariableop_37_conv2_c5_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_38Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_39IdentityIdentity_38:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_39Identity_39:output:0*a
_input_shapesP
N: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
n
B__inference_off_cat1_layer_call_and_return_conditional_losses_4401
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :|
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*,
_output_shapes
:���������� \
IdentityIdentityconcat:output:0*
T0*,
_output_shapes
:���������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:���������� :���������� :V R
,
_output_shapes
:���������� 
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:���������� 
"
_user_specified_name
inputs_1
�
�
?__inference_bn_c3_layer_call_and_return_conditional_losses_4729

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
J
.__inference_max_pooling2d_3_layer_call_fn_4828

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1295�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
y
A__inference_offsets_layer_call_and_return_conditional_losses_2268

inputs
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*,
_output_shapes
:����������*\
IdentityIdentityconcat:output:0*
T0*,
_output_shapes
:����������*"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������� :����������:����������:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
$__inference_model_layer_call_fn_1503
	input_map!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17:  

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23: 

unknown_24:
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	input_mapunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:���������   :��������� :���������*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_1444w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������   y

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:��������� y

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
1
_output_shapes
:�����������
#
_user_specified_name	input_map
��
�
B__inference_ssd_head_layer_call_and_return_conditional_losses_3778

inputsG
-model_conv2_c1_conv2d_readvariableop_resource:<
.model_conv2_c1_biasadd_readvariableop_resource:1
#model_bn_c1_readvariableop_resource:3
%model_bn_c1_readvariableop_1_resource:B
4model_bn_c1_fusedbatchnormv3_readvariableop_resource:D
6model_bn_c1_fusedbatchnormv3_readvariableop_1_resource:G
-model_conv2_c2_conv2d_readvariableop_resource:<
.model_conv2_c2_biasadd_readvariableop_resource:1
#model_bn_c2_readvariableop_resource:3
%model_bn_c2_readvariableop_1_resource:B
4model_bn_c2_fusedbatchnormv3_readvariableop_resource:D
6model_bn_c2_fusedbatchnormv3_readvariableop_1_resource:G
-model_conv2_c3_conv2d_readvariableop_resource: <
.model_conv2_c3_biasadd_readvariableop_resource: 1
#model_bn_c3_readvariableop_resource: 3
%model_bn_c3_readvariableop_1_resource: B
4model_bn_c3_fusedbatchnormv3_readvariableop_resource: D
6model_bn_c3_fusedbatchnormv3_readvariableop_1_resource: G
-model_conv2_c4_conv2d_readvariableop_resource:  <
.model_conv2_c4_biasadd_readvariableop_resource: 1
#model_bn_c4_readvariableop_resource: 3
%model_bn_c4_readvariableop_1_resource: B
4model_bn_c4_fusedbatchnormv3_readvariableop_resource: D
6model_bn_c4_fusedbatchnormv3_readvariableop_1_resource: G
-model_conv2_c5_conv2d_readvariableop_resource: <
.model_conv2_c5_biasadd_readvariableop_resource:=
#off3_conv2d_readvariableop_resource:2
$off3_biasadd_readvariableop_resource:=
#off2_conv2d_readvariableop_resource: 2
$off2_biasadd_readvariableop_resource:=
#off1_conv2d_readvariableop_resource: 2
$off1_biasadd_readvariableop_resource:=
#cls3_conv2d_readvariableop_resource:2
$cls3_biasadd_readvariableop_resource:=
#cls2_conv2d_readvariableop_resource: 2
$cls2_biasadd_readvariableop_resource:=
#cls1_conv2d_readvariableop_resource: 2
$cls1_biasadd_readvariableop_resource:
identity

identity_1��cls1/BiasAdd/ReadVariableOp�cls1/Conv2D/ReadVariableOp�cls2/BiasAdd/ReadVariableOp�cls2/Conv2D/ReadVariableOp�cls3/BiasAdd/ReadVariableOp�cls3/Conv2D/ReadVariableOp�model/bn_c1/AssignNewValue�model/bn_c1/AssignNewValue_1�+model/bn_c1/FusedBatchNormV3/ReadVariableOp�-model/bn_c1/FusedBatchNormV3/ReadVariableOp_1�model/bn_c1/ReadVariableOp�model/bn_c1/ReadVariableOp_1�model/bn_c2/AssignNewValue�model/bn_c2/AssignNewValue_1�+model/bn_c2/FusedBatchNormV3/ReadVariableOp�-model/bn_c2/FusedBatchNormV3/ReadVariableOp_1�model/bn_c2/ReadVariableOp�model/bn_c2/ReadVariableOp_1�model/bn_c3/AssignNewValue�model/bn_c3/AssignNewValue_1�+model/bn_c3/FusedBatchNormV3/ReadVariableOp�-model/bn_c3/FusedBatchNormV3/ReadVariableOp_1�model/bn_c3/ReadVariableOp�model/bn_c3/ReadVariableOp_1�model/bn_c4/AssignNewValue�model/bn_c4/AssignNewValue_1�+model/bn_c4/FusedBatchNormV3/ReadVariableOp�-model/bn_c4/FusedBatchNormV3/ReadVariableOp_1�model/bn_c4/ReadVariableOp�model/bn_c4/ReadVariableOp_1�%model/conv2_c1/BiasAdd/ReadVariableOp�$model/conv2_c1/Conv2D/ReadVariableOp�%model/conv2_c2/BiasAdd/ReadVariableOp�$model/conv2_c2/Conv2D/ReadVariableOp�%model/conv2_c3/BiasAdd/ReadVariableOp�$model/conv2_c3/Conv2D/ReadVariableOp�%model/conv2_c4/BiasAdd/ReadVariableOp�$model/conv2_c4/Conv2D/ReadVariableOp�%model/conv2_c5/BiasAdd/ReadVariableOp�$model/conv2_c5/Conv2D/ReadVariableOp�off1/BiasAdd/ReadVariableOp�off1/Conv2D/ReadVariableOp�off2/BiasAdd/ReadVariableOp�off2/Conv2D/ReadVariableOp�off3/BiasAdd/ReadVariableOp�off3/Conv2D/ReadVariableOpn
model/conv2_c1/CastCastinputs*

DstT0*

SrcT0*1
_output_shapes
:������������
$model/conv2_c1/Conv2D/ReadVariableOpReadVariableOp-model_conv2_c1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
model/conv2_c1/Conv2D/CastCast,model/conv2_c1/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
model/conv2_c1/Conv2DConv2Dmodel/conv2_c1/Cast:y:0model/conv2_c1/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
%model/conv2_c1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2_c1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv2_c1/BiasAdd/CastCast-model/conv2_c1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
model/conv2_c1/BiasAddBiasAddmodel/conv2_c1/Conv2D:output:0model/conv2_c1/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������x
model/conv2_c1/ReluRelumodel/conv2_c1/BiasAdd:output:0*
T0*1
_output_shapes
:�����������z
model/bn_c1/ReadVariableOpReadVariableOp#model_bn_c1_readvariableop_resource*
_output_shapes
:*
dtype0~
model/bn_c1/ReadVariableOp_1ReadVariableOp%model_bn_c1_readvariableop_1_resource*
_output_shapes
:*
dtype0�
+model/bn_c1/FusedBatchNormV3/ReadVariableOpReadVariableOp4model_bn_c1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
-model/bn_c1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6model_bn_c1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
model/bn_c1/FusedBatchNormV3FusedBatchNormV3!model/conv2_c1/Relu:activations:0"model/bn_c1/ReadVariableOp:value:0$model/bn_c1/ReadVariableOp_1:value:03model/bn_c1/FusedBatchNormV3/ReadVariableOp:value:05model/bn_c1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:�����������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
model/bn_c1/AssignNewValueAssignVariableOp4model_bn_c1_fusedbatchnormv3_readvariableop_resource)model/bn_c1/FusedBatchNormV3:batch_mean:0,^model/bn_c1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
model/bn_c1/AssignNewValue_1AssignVariableOp6model_bn_c1_fusedbatchnormv3_readvariableop_1_resource-model/bn_c1/FusedBatchNormV3:batch_variance:0.^model/bn_c1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
model/max_pooling2d/MaxPoolMaxPool model/bn_c1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@@*
ksize
*
paddingSAME*
strides
�
$model/conv2_c2/Conv2D/ReadVariableOpReadVariableOp-model_conv2_c2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
model/conv2_c2/Conv2D/CastCast,model/conv2_c2/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
model/conv2_c2/Conv2DConv2D$model/max_pooling2d/MaxPool:output:0model/conv2_c2/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
�
%model/conv2_c2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2_c2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv2_c2/BiasAdd/CastCast-model/conv2_c2/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
model/conv2_c2/BiasAddBiasAddmodel/conv2_c2/Conv2D:output:0model/conv2_c2/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������@@v
model/conv2_c2/ReluRelumodel/conv2_c2/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@z
model/bn_c2/ReadVariableOpReadVariableOp#model_bn_c2_readvariableop_resource*
_output_shapes
:*
dtype0~
model/bn_c2/ReadVariableOp_1ReadVariableOp%model_bn_c2_readvariableop_1_resource*
_output_shapes
:*
dtype0�
+model/bn_c2/FusedBatchNormV3/ReadVariableOpReadVariableOp4model_bn_c2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
-model/bn_c2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6model_bn_c2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
model/bn_c2/FusedBatchNormV3FusedBatchNormV3!model/conv2_c2/Relu:activations:0"model/bn_c2/ReadVariableOp:value:0$model/bn_c2/ReadVariableOp_1:value:03model/bn_c2/FusedBatchNormV3/ReadVariableOp:value:05model/bn_c2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@@:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
model/bn_c2/AssignNewValueAssignVariableOp4model_bn_c2_fusedbatchnormv3_readvariableop_resource)model/bn_c2/FusedBatchNormV3:batch_mean:0,^model/bn_c2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
model/bn_c2/AssignNewValue_1AssignVariableOp6model_bn_c2_fusedbatchnormv3_readvariableop_1_resource-model/bn_c2/FusedBatchNormV3:batch_variance:0.^model/bn_c2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
model/max_pooling2d_1/MaxPoolMaxPool model/bn_c2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������  *
ksize
*
paddingSAME*
strides
�
$model/conv2_c3/Conv2D/ReadVariableOpReadVariableOp-model_conv2_c3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
model/conv2_c3/Conv2D/CastCast,model/conv2_c3/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
model/conv2_c3/Conv2DConv2D&model/max_pooling2d_1/MaxPool:output:0model/conv2_c3/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
�
%model/conv2_c3/BiasAdd/ReadVariableOpReadVariableOp.model_conv2_c3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model/conv2_c3/BiasAdd/CastCast-model/conv2_c3/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: �
model/conv2_c3/BiasAddBiasAddmodel/conv2_c3/Conv2D:output:0model/conv2_c3/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������   v
model/conv2_c3/ReluRelumodel/conv2_c3/BiasAdd:output:0*
T0*/
_output_shapes
:���������   z
model/bn_c3/ReadVariableOpReadVariableOp#model_bn_c3_readvariableop_resource*
_output_shapes
: *
dtype0~
model/bn_c3/ReadVariableOp_1ReadVariableOp%model_bn_c3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
+model/bn_c3/FusedBatchNormV3/ReadVariableOpReadVariableOp4model_bn_c3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
-model/bn_c3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6model_bn_c3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
model/bn_c3/FusedBatchNormV3FusedBatchNormV3!model/conv2_c3/Relu:activations:0"model/bn_c3/ReadVariableOp:value:0$model/bn_c3/ReadVariableOp_1:value:03model/bn_c3/FusedBatchNormV3/ReadVariableOp:value:05model/bn_c3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
model/bn_c3/AssignNewValueAssignVariableOp4model_bn_c3_fusedbatchnormv3_readvariableop_resource)model/bn_c3/FusedBatchNormV3:batch_mean:0,^model/bn_c3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
model/bn_c3/AssignNewValue_1AssignVariableOp6model_bn_c3_fusedbatchnormv3_readvariableop_1_resource-model/bn_c3/FusedBatchNormV3:batch_variance:0.^model/bn_c3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
model/max_pooling2d_2/MaxPoolMaxPool model/bn_c3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:��������� *
ksize
*
paddingSAME*
strides
�
$model/conv2_c4/Conv2D/ReadVariableOpReadVariableOp-model_conv2_c4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
model/conv2_c4/Conv2D/CastCast,model/conv2_c4/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:  �
model/conv2_c4/Conv2DConv2D&model/max_pooling2d_2/MaxPool:output:0model/conv2_c4/Conv2D/Cast:y:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
%model/conv2_c4/BiasAdd/ReadVariableOpReadVariableOp.model_conv2_c4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model/conv2_c4/BiasAdd/CastCast-model/conv2_c4/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: �
model/conv2_c4/BiasAddBiasAddmodel/conv2_c4/Conv2D:output:0model/conv2_c4/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:��������� v
model/conv2_c4/ReluRelumodel/conv2_c4/BiasAdd:output:0*
T0*/
_output_shapes
:��������� z
model/bn_c4/ReadVariableOpReadVariableOp#model_bn_c4_readvariableop_resource*
_output_shapes
: *
dtype0~
model/bn_c4/ReadVariableOp_1ReadVariableOp%model_bn_c4_readvariableop_1_resource*
_output_shapes
: *
dtype0�
+model/bn_c4/FusedBatchNormV3/ReadVariableOpReadVariableOp4model_bn_c4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
-model/bn_c4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6model_bn_c4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
model/bn_c4/FusedBatchNormV3FusedBatchNormV3!model/conv2_c4/Relu:activations:0"model/bn_c4/ReadVariableOp:value:0$model/bn_c4/ReadVariableOp_1:value:03model/bn_c4/FusedBatchNormV3/ReadVariableOp:value:05model/bn_c4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
model/bn_c4/AssignNewValueAssignVariableOp4model_bn_c4_fusedbatchnormv3_readvariableop_resource)model/bn_c4/FusedBatchNormV3:batch_mean:0,^model/bn_c4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
model/bn_c4/AssignNewValue_1AssignVariableOp6model_bn_c4_fusedbatchnormv3_readvariableop_1_resource-model/bn_c4/FusedBatchNormV3:batch_variance:0.^model/bn_c4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
model/max_pooling2d_3/MaxPoolMaxPool model/bn_c4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:��������� *
ksize
*
paddingSAME*
strides
�
$model/conv2_c5/Conv2D/ReadVariableOpReadVariableOp-model_conv2_c5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
model/conv2_c5/Conv2D/CastCast,model/conv2_c5/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
model/conv2_c5/Conv2DConv2D&model/max_pooling2d_3/MaxPool:output:0model/conv2_c5/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
%model/conv2_c5/BiasAdd/ReadVariableOpReadVariableOp.model_conv2_c5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv2_c5/BiasAdd/CastCast-model/conv2_c5/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
model/conv2_c5/BiasAddBiasAddmodel/conv2_c5/Conv2D:output:0model/conv2_c5/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������v
model/conv2_c5/ReluRelumodel/conv2_c5/BiasAdd:output:0*
T0*/
_output_shapes
:����������
off3/Conv2D/ReadVariableOpReadVariableOp#off3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0|
off3/Conv2D/CastCast"off3/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
off3/Conv2DConv2D!model/conv2_c5/Relu:activations:0off3/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
|
off3/BiasAdd/ReadVariableOpReadVariableOp$off3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0r
off3/BiasAdd/CastCast#off3/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:~
off3/BiasAddBiasAddoff3/Conv2D:output:0off3/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:����������
off2/Conv2D/ReadVariableOpReadVariableOp#off2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0|
off2/Conv2D/CastCast"off2/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
off2/Conv2DConv2D model/bn_c4/FusedBatchNormV3:y:0off2/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
|
off2/BiasAdd/ReadVariableOpReadVariableOp$off2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0r
off2/BiasAdd/CastCast#off2/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:~
off2/BiasAddBiasAddoff2/Conv2D:output:0off2/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:����������
off1/Conv2D/ReadVariableOpReadVariableOp#off1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0|
off1/Conv2D/CastCast"off1/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
off1/Conv2DConv2D model/bn_c3/FusedBatchNormV3:y:0off1/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
|
off1/BiasAdd/ReadVariableOpReadVariableOp$off1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0r
off1/BiasAdd/CastCast#off1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:~
off1/BiasAddBiasAddoff1/Conv2D:output:0off1/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������  �
cls3/Conv2D/ReadVariableOpReadVariableOp#cls3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0|
cls3/Conv2D/CastCast"cls3/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
cls3/Conv2DConv2D!model/conv2_c5/Relu:activations:0cls3/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
|
cls3/BiasAdd/ReadVariableOpReadVariableOp$cls3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0r
cls3/BiasAdd/CastCast#cls3/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:~
cls3/BiasAddBiasAddcls3/Conv2D:output:0cls3/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:����������
cls2/Conv2D/ReadVariableOpReadVariableOp#cls2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0|
cls2/Conv2D/CastCast"cls2/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
cls2/Conv2DConv2D model/bn_c4/FusedBatchNormV3:y:0cls2/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
|
cls2/BiasAdd/ReadVariableOpReadVariableOp$cls2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0r
cls2/BiasAdd/CastCast#cls2/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:~
cls2/BiasAddBiasAddcls2/Conv2D:output:0cls2/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:����������
cls1/Conv2D/ReadVariableOpReadVariableOp#cls1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0|
cls1/Conv2D/CastCast"cls1/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
cls1/Conv2DConv2D model/bn_c3/FusedBatchNormV3:y:0cls1/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
|
cls1/BiasAdd/ReadVariableOpReadVariableOp$cls1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0r
cls1/BiasAdd/CastCast#cls1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:~
cls1/BiasAddBiasAddcls1/Conv2D:output:0cls1/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������  S
off_res3/ShapeShapeoff3/BiasAdd:output:0*
T0*
_output_shapes
:f
off_res3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
off_res3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
off_res3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
off_res3/strided_sliceStridedSliceoff_res3/Shape:output:0%off_res3/strided_slice/stack:output:0'off_res3/strided_slice/stack_1:output:0'off_res3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
off_res3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������Z
off_res3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
off_res3/Reshape/shapePackoff_res3/strided_slice:output:0!off_res3/Reshape/shape/1:output:0!off_res3/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
off_res3/ReshapeReshapeoff3/BiasAdd:output:0off_res3/Reshape/shape:output:0*
T0*,
_output_shapes
:����������S
off_res2/ShapeShapeoff2/BiasAdd:output:0*
T0*
_output_shapes
:f
off_res2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
off_res2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
off_res2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
off_res2/strided_sliceStridedSliceoff_res2/Shape:output:0%off_res2/strided_slice/stack:output:0'off_res2/strided_slice/stack_1:output:0'off_res2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
off_res2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������Z
off_res2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
off_res2/Reshape/shapePackoff_res2/strided_slice:output:0!off_res2/Reshape/shape/1:output:0!off_res2/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
off_res2/ReshapeReshapeoff2/BiasAdd:output:0off_res2/Reshape/shape:output:0*
T0*,
_output_shapes
:����������S
off_res1/ShapeShapeoff1/BiasAdd:output:0*
T0*
_output_shapes
:f
off_res1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
off_res1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
off_res1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
off_res1/strided_sliceStridedSliceoff_res1/Shape:output:0%off_res1/strided_slice/stack:output:0'off_res1/strided_slice/stack_1:output:0'off_res1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
off_res1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������Z
off_res1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
off_res1/Reshape/shapePackoff_res1/strided_slice:output:0!off_res1/Reshape/shape/1:output:0!off_res1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
off_res1/ReshapeReshapeoff1/BiasAdd:output:0off_res1/Reshape/shape:output:0*
T0*,
_output_shapes
:���������� S
cls_res3/ShapeShapecls3/BiasAdd:output:0*
T0*
_output_shapes
:f
cls_res3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
cls_res3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
cls_res3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
cls_res3/strided_sliceStridedSlicecls_res3/Shape:output:0%cls_res3/strided_slice/stack:output:0'cls_res3/strided_slice/stack_1:output:0'cls_res3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
cls_res3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������Z
cls_res3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
cls_res3/Reshape/shapePackcls_res3/strided_slice:output:0!cls_res3/Reshape/shape/1:output:0!cls_res3/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
cls_res3/ReshapeReshapecls3/BiasAdd:output:0cls_res3/Reshape/shape:output:0*
T0*,
_output_shapes
:����������S
cls_res2/ShapeShapecls2/BiasAdd:output:0*
T0*
_output_shapes
:f
cls_res2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
cls_res2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
cls_res2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
cls_res2/strided_sliceStridedSlicecls_res2/Shape:output:0%cls_res2/strided_slice/stack:output:0'cls_res2/strided_slice/stack_1:output:0'cls_res2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
cls_res2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������Z
cls_res2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
cls_res2/Reshape/shapePackcls_res2/strided_slice:output:0!cls_res2/Reshape/shape/1:output:0!cls_res2/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
cls_res2/ReshapeReshapecls2/BiasAdd:output:0cls_res2/Reshape/shape:output:0*
T0*,
_output_shapes
:����������S
cls_res1/ShapeShapecls1/BiasAdd:output:0*
T0*
_output_shapes
:f
cls_res1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
cls_res1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
cls_res1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
cls_res1/strided_sliceStridedSlicecls_res1/Shape:output:0%cls_res1/strided_slice/stack:output:0'cls_res1/strided_slice/stack_1:output:0'cls_res1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
cls_res1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������Z
cls_res1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
cls_res1/Reshape/shapePackcls_res1/strided_slice:output:0!cls_res1/Reshape/shape/1:output:0!cls_res1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
cls_res1/ReshapeReshapecls1/BiasAdd:output:0cls_res1/Reshape/shape:output:0*
T0*,
_output_shapes
:���������� V
off_cat1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
off_cat1/concatConcatV2off_res1/Reshape:output:0off_res1/Reshape:output:0off_cat1/concat/axis:output:0*
N*
T0*,
_output_shapes
:���������� V
off_cat2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
off_cat2/concatConcatV2off_res2/Reshape:output:0off_res2/Reshape:output:0off_cat2/concat/axis:output:0*
N*
T0*,
_output_shapes
:����������V
off_cat3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
off_cat3/concatConcatV2off_res3/Reshape:output:0off_res3/Reshape:output:0off_cat3/concat/axis:output:0*
N*
T0*,
_output_shapes
:����������m
cls_out1/SoftmaxSoftmaxcls_res1/Reshape:output:0*
T0*,
_output_shapes
:���������� m
cls_out2/SoftmaxSoftmaxcls_res2/Reshape:output:0*
T0*,
_output_shapes
:����������m
cls_out3/SoftmaxSoftmaxcls_res3/Reshape:output:0*
T0*,
_output_shapes
:����������U
offsets/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
offsets/concatConcatV2off_cat1/concat:output:0off_cat2/concat:output:0off_cat3/concat:output:0offsets/concat/axis:output:0*
N*
T0*,
_output_shapes
:����������*U
classes/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
classes/concatConcatV2cls_out1/Softmax:softmax:0cls_out2/Softmax:softmax:0cls_out3/Softmax:softmax:0classes/concat/axis:output:0*
N*
T0*,
_output_shapes
:����������*k
IdentityIdentityclasses/concat:output:0^NoOp*
T0*,
_output_shapes
:����������*m

Identity_1Identityoffsets/concat:output:0^NoOp*
T0*,
_output_shapes
:����������*�
NoOpNoOp^cls1/BiasAdd/ReadVariableOp^cls1/Conv2D/ReadVariableOp^cls2/BiasAdd/ReadVariableOp^cls2/Conv2D/ReadVariableOp^cls3/BiasAdd/ReadVariableOp^cls3/Conv2D/ReadVariableOp^model/bn_c1/AssignNewValue^model/bn_c1/AssignNewValue_1,^model/bn_c1/FusedBatchNormV3/ReadVariableOp.^model/bn_c1/FusedBatchNormV3/ReadVariableOp_1^model/bn_c1/ReadVariableOp^model/bn_c1/ReadVariableOp_1^model/bn_c2/AssignNewValue^model/bn_c2/AssignNewValue_1,^model/bn_c2/FusedBatchNormV3/ReadVariableOp.^model/bn_c2/FusedBatchNormV3/ReadVariableOp_1^model/bn_c2/ReadVariableOp^model/bn_c2/ReadVariableOp_1^model/bn_c3/AssignNewValue^model/bn_c3/AssignNewValue_1,^model/bn_c3/FusedBatchNormV3/ReadVariableOp.^model/bn_c3/FusedBatchNormV3/ReadVariableOp_1^model/bn_c3/ReadVariableOp^model/bn_c3/ReadVariableOp_1^model/bn_c4/AssignNewValue^model/bn_c4/AssignNewValue_1,^model/bn_c4/FusedBatchNormV3/ReadVariableOp.^model/bn_c4/FusedBatchNormV3/ReadVariableOp_1^model/bn_c4/ReadVariableOp^model/bn_c4/ReadVariableOp_1&^model/conv2_c1/BiasAdd/ReadVariableOp%^model/conv2_c1/Conv2D/ReadVariableOp&^model/conv2_c2/BiasAdd/ReadVariableOp%^model/conv2_c2/Conv2D/ReadVariableOp&^model/conv2_c3/BiasAdd/ReadVariableOp%^model/conv2_c3/Conv2D/ReadVariableOp&^model/conv2_c4/BiasAdd/ReadVariableOp%^model/conv2_c4/Conv2D/ReadVariableOp&^model/conv2_c5/BiasAdd/ReadVariableOp%^model/conv2_c5/Conv2D/ReadVariableOp^off1/BiasAdd/ReadVariableOp^off1/Conv2D/ReadVariableOp^off2/BiasAdd/ReadVariableOp^off2/Conv2D/ReadVariableOp^off3/BiasAdd/ReadVariableOp^off3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
cls1/BiasAdd/ReadVariableOpcls1/BiasAdd/ReadVariableOp28
cls1/Conv2D/ReadVariableOpcls1/Conv2D/ReadVariableOp2:
cls2/BiasAdd/ReadVariableOpcls2/BiasAdd/ReadVariableOp28
cls2/Conv2D/ReadVariableOpcls2/Conv2D/ReadVariableOp2:
cls3/BiasAdd/ReadVariableOpcls3/BiasAdd/ReadVariableOp28
cls3/Conv2D/ReadVariableOpcls3/Conv2D/ReadVariableOp28
model/bn_c1/AssignNewValuemodel/bn_c1/AssignNewValue2<
model/bn_c1/AssignNewValue_1model/bn_c1/AssignNewValue_12Z
+model/bn_c1/FusedBatchNormV3/ReadVariableOp+model/bn_c1/FusedBatchNormV3/ReadVariableOp2^
-model/bn_c1/FusedBatchNormV3/ReadVariableOp_1-model/bn_c1/FusedBatchNormV3/ReadVariableOp_128
model/bn_c1/ReadVariableOpmodel/bn_c1/ReadVariableOp2<
model/bn_c1/ReadVariableOp_1model/bn_c1/ReadVariableOp_128
model/bn_c2/AssignNewValuemodel/bn_c2/AssignNewValue2<
model/bn_c2/AssignNewValue_1model/bn_c2/AssignNewValue_12Z
+model/bn_c2/FusedBatchNormV3/ReadVariableOp+model/bn_c2/FusedBatchNormV3/ReadVariableOp2^
-model/bn_c2/FusedBatchNormV3/ReadVariableOp_1-model/bn_c2/FusedBatchNormV3/ReadVariableOp_128
model/bn_c2/ReadVariableOpmodel/bn_c2/ReadVariableOp2<
model/bn_c2/ReadVariableOp_1model/bn_c2/ReadVariableOp_128
model/bn_c3/AssignNewValuemodel/bn_c3/AssignNewValue2<
model/bn_c3/AssignNewValue_1model/bn_c3/AssignNewValue_12Z
+model/bn_c3/FusedBatchNormV3/ReadVariableOp+model/bn_c3/FusedBatchNormV3/ReadVariableOp2^
-model/bn_c3/FusedBatchNormV3/ReadVariableOp_1-model/bn_c3/FusedBatchNormV3/ReadVariableOp_128
model/bn_c3/ReadVariableOpmodel/bn_c3/ReadVariableOp2<
model/bn_c3/ReadVariableOp_1model/bn_c3/ReadVariableOp_128
model/bn_c4/AssignNewValuemodel/bn_c4/AssignNewValue2<
model/bn_c4/AssignNewValue_1model/bn_c4/AssignNewValue_12Z
+model/bn_c4/FusedBatchNormV3/ReadVariableOp+model/bn_c4/FusedBatchNormV3/ReadVariableOp2^
-model/bn_c4/FusedBatchNormV3/ReadVariableOp_1-model/bn_c4/FusedBatchNormV3/ReadVariableOp_128
model/bn_c4/ReadVariableOpmodel/bn_c4/ReadVariableOp2<
model/bn_c4/ReadVariableOp_1model/bn_c4/ReadVariableOp_12N
%model/conv2_c1/BiasAdd/ReadVariableOp%model/conv2_c1/BiasAdd/ReadVariableOp2L
$model/conv2_c1/Conv2D/ReadVariableOp$model/conv2_c1/Conv2D/ReadVariableOp2N
%model/conv2_c2/BiasAdd/ReadVariableOp%model/conv2_c2/BiasAdd/ReadVariableOp2L
$model/conv2_c2/Conv2D/ReadVariableOp$model/conv2_c2/Conv2D/ReadVariableOp2N
%model/conv2_c3/BiasAdd/ReadVariableOp%model/conv2_c3/BiasAdd/ReadVariableOp2L
$model/conv2_c3/Conv2D/ReadVariableOp$model/conv2_c3/Conv2D/ReadVariableOp2N
%model/conv2_c4/BiasAdd/ReadVariableOp%model/conv2_c4/BiasAdd/ReadVariableOp2L
$model/conv2_c4/Conv2D/ReadVariableOp$model/conv2_c4/Conv2D/ReadVariableOp2N
%model/conv2_c5/BiasAdd/ReadVariableOp%model/conv2_c5/BiasAdd/ReadVariableOp2L
$model/conv2_c5/Conv2D/ReadVariableOp$model/conv2_c5/Conv2D/ReadVariableOp2:
off1/BiasAdd/ReadVariableOpoff1/BiasAdd/ReadVariableOp28
off1/Conv2D/ReadVariableOpoff1/Conv2D/ReadVariableOp2:
off2/BiasAdd/ReadVariableOpoff2/BiasAdd/ReadVariableOp28
off2/Conv2D/ReadVariableOpoff2/Conv2D/ReadVariableOp2:
off3/BiasAdd/ReadVariableOpoff3/BiasAdd/ReadVariableOp28
off3/Conv2D/ReadVariableOpoff3/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
^
B__inference_cls_out1_layer_call_and_return_conditional_losses_4368

inputs
identityQ
SoftmaxSoftmaxinputs*
T0*,
_output_shapes
:���������� ^
IdentityIdentitySoftmax:softmax:0*
T0*,
_output_shapes
:���������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
�	
'__inference_ssd_head_layer_call_fn_3326

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17:  

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23: 

unknown_24:$

unknown_25:

unknown_26:$

unknown_27: 

unknown_28:$

unknown_29: 

unknown_30:$

unknown_31:

unknown_32:$

unknown_33: 

unknown_34:$

unknown_35: 

unknown_36:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *D
_output_shapes2
0:����������*:����������**@
_read_only_resource_inputs"
 	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_ssd_head_layer_call_and_return_conditional_losses_2703t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������*v

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*,
_output_shapes
:����������*`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�

^
B__inference_off_res1_layer_call_and_return_conditional_losses_2165

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:i
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:���������� ]
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:���������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������  :W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
n
B__inference_off_cat3_layer_call_and_return_conditional_losses_4427
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :|
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*,
_output_shapes
:����������\
IdentityIdentityconcat:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������:����������:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs_1
�
�
>__inference_cls3_layer_call_and_return_conditional_losses_4187

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:o
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
$__inference_bn_c3_layer_call_fn_4680

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_bn_c3_layer_call_and_return_conditional_losses_1168�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
C
'__inference_off_res3_layer_call_fn_4345

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_off_res3_layer_call_and_return_conditional_losses_2135e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
B__inference_conv2_c5_layer_call_and_return_conditional_losses_1435

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:o
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
C
'__inference_cls_out3_layer_call_fn_4383

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_cls_out3_layer_call_and_return_conditional_losses_2258e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
>__inference_cls3_layer_call_and_return_conditional_losses_2080

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:o
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
`
&__inference_classes_layer_call_fn_4434
inputs_0
inputs_1
inputs_2
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_classes_layer_call_and_return_conditional_losses_2278e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������*"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������� :����������:����������:V R
,
_output_shapes
:���������� 
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs_1:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs_2
�
�
$__inference_bn_c2_layer_call_fn_4599

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_bn_c2_layer_call_and_return_conditional_losses_1123�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1067

inputs
identity�
MaxPoolMaxPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
$__inference_model_layer_call_fn_1808
	input_map!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17:  

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23: 

unknown_24:
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	input_mapunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:���������   :��������� :���������*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_1688w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������   y

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:��������� y

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
1
_output_shapes
:�����������
#
_user_specified_name	input_map
�
l
B__inference_off_cat1_layer_call_and_return_conditional_losses_2219

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :z
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*,
_output_shapes
:���������� \
IdentityIdentityconcat:output:0*
T0*,
_output_shapes
:���������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:���������� :���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs:TP
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
�
$__inference_bn_c4_layer_call_fn_4774

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_bn_c4_layer_call_and_return_conditional_losses_1244�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
?__inference_bn_c4_layer_call_and_return_conditional_losses_1275

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
#__inference_cls1_layer_call_fn_4133

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_cls1_layer_call_and_return_conditional_losses_2116w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
�
?__inference_bn_c2_layer_call_and_return_conditional_losses_1092

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
#__inference_off2_layer_call_fn_4217

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_off2_layer_call_and_return_conditional_losses_2044w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_4739

inputs
identity�
MaxPoolMaxPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

^
B__inference_cls_res2_layer_call_and_return_conditional_losses_2195

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:i
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:����������]
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
��
�"
__inference__wrapped_model_994
input_1P
6ssd_head_model_conv2_c1_conv2d_readvariableop_resource:E
7ssd_head_model_conv2_c1_biasadd_readvariableop_resource::
,ssd_head_model_bn_c1_readvariableop_resource:<
.ssd_head_model_bn_c1_readvariableop_1_resource:K
=ssd_head_model_bn_c1_fusedbatchnormv3_readvariableop_resource:M
?ssd_head_model_bn_c1_fusedbatchnormv3_readvariableop_1_resource:P
6ssd_head_model_conv2_c2_conv2d_readvariableop_resource:E
7ssd_head_model_conv2_c2_biasadd_readvariableop_resource::
,ssd_head_model_bn_c2_readvariableop_resource:<
.ssd_head_model_bn_c2_readvariableop_1_resource:K
=ssd_head_model_bn_c2_fusedbatchnormv3_readvariableop_resource:M
?ssd_head_model_bn_c2_fusedbatchnormv3_readvariableop_1_resource:P
6ssd_head_model_conv2_c3_conv2d_readvariableop_resource: E
7ssd_head_model_conv2_c3_biasadd_readvariableop_resource: :
,ssd_head_model_bn_c3_readvariableop_resource: <
.ssd_head_model_bn_c3_readvariableop_1_resource: K
=ssd_head_model_bn_c3_fusedbatchnormv3_readvariableop_resource: M
?ssd_head_model_bn_c3_fusedbatchnormv3_readvariableop_1_resource: P
6ssd_head_model_conv2_c4_conv2d_readvariableop_resource:  E
7ssd_head_model_conv2_c4_biasadd_readvariableop_resource: :
,ssd_head_model_bn_c4_readvariableop_resource: <
.ssd_head_model_bn_c4_readvariableop_1_resource: K
=ssd_head_model_bn_c4_fusedbatchnormv3_readvariableop_resource: M
?ssd_head_model_bn_c4_fusedbatchnormv3_readvariableop_1_resource: P
6ssd_head_model_conv2_c5_conv2d_readvariableop_resource: E
7ssd_head_model_conv2_c5_biasadd_readvariableop_resource:F
,ssd_head_off3_conv2d_readvariableop_resource:;
-ssd_head_off3_biasadd_readvariableop_resource:F
,ssd_head_off2_conv2d_readvariableop_resource: ;
-ssd_head_off2_biasadd_readvariableop_resource:F
,ssd_head_off1_conv2d_readvariableop_resource: ;
-ssd_head_off1_biasadd_readvariableop_resource:F
,ssd_head_cls3_conv2d_readvariableop_resource:;
-ssd_head_cls3_biasadd_readvariableop_resource:F
,ssd_head_cls2_conv2d_readvariableop_resource: ;
-ssd_head_cls2_biasadd_readvariableop_resource:F
,ssd_head_cls1_conv2d_readvariableop_resource: ;
-ssd_head_cls1_biasadd_readvariableop_resource:
identity

identity_1��$ssd_head/cls1/BiasAdd/ReadVariableOp�#ssd_head/cls1/Conv2D/ReadVariableOp�$ssd_head/cls2/BiasAdd/ReadVariableOp�#ssd_head/cls2/Conv2D/ReadVariableOp�$ssd_head/cls3/BiasAdd/ReadVariableOp�#ssd_head/cls3/Conv2D/ReadVariableOp�4ssd_head/model/bn_c1/FusedBatchNormV3/ReadVariableOp�6ssd_head/model/bn_c1/FusedBatchNormV3/ReadVariableOp_1�#ssd_head/model/bn_c1/ReadVariableOp�%ssd_head/model/bn_c1/ReadVariableOp_1�4ssd_head/model/bn_c2/FusedBatchNormV3/ReadVariableOp�6ssd_head/model/bn_c2/FusedBatchNormV3/ReadVariableOp_1�#ssd_head/model/bn_c2/ReadVariableOp�%ssd_head/model/bn_c2/ReadVariableOp_1�4ssd_head/model/bn_c3/FusedBatchNormV3/ReadVariableOp�6ssd_head/model/bn_c3/FusedBatchNormV3/ReadVariableOp_1�#ssd_head/model/bn_c3/ReadVariableOp�%ssd_head/model/bn_c3/ReadVariableOp_1�4ssd_head/model/bn_c4/FusedBatchNormV3/ReadVariableOp�6ssd_head/model/bn_c4/FusedBatchNormV3/ReadVariableOp_1�#ssd_head/model/bn_c4/ReadVariableOp�%ssd_head/model/bn_c4/ReadVariableOp_1�.ssd_head/model/conv2_c1/BiasAdd/ReadVariableOp�-ssd_head/model/conv2_c1/Conv2D/ReadVariableOp�.ssd_head/model/conv2_c2/BiasAdd/ReadVariableOp�-ssd_head/model/conv2_c2/Conv2D/ReadVariableOp�.ssd_head/model/conv2_c3/BiasAdd/ReadVariableOp�-ssd_head/model/conv2_c3/Conv2D/ReadVariableOp�.ssd_head/model/conv2_c4/BiasAdd/ReadVariableOp�-ssd_head/model/conv2_c4/Conv2D/ReadVariableOp�.ssd_head/model/conv2_c5/BiasAdd/ReadVariableOp�-ssd_head/model/conv2_c5/Conv2D/ReadVariableOp�$ssd_head/off1/BiasAdd/ReadVariableOp�#ssd_head/off1/Conv2D/ReadVariableOp�$ssd_head/off2/BiasAdd/ReadVariableOp�#ssd_head/off2/Conv2D/ReadVariableOp�$ssd_head/off3/BiasAdd/ReadVariableOp�#ssd_head/off3/Conv2D/ReadVariableOpx
ssd_head/model/conv2_c1/CastCastinput_1*

DstT0*

SrcT0*1
_output_shapes
:������������
-ssd_head/model/conv2_c1/Conv2D/ReadVariableOpReadVariableOp6ssd_head_model_conv2_c1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
#ssd_head/model/conv2_c1/Conv2D/CastCast5ssd_head/model/conv2_c1/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
ssd_head/model/conv2_c1/Conv2DConv2D ssd_head/model/conv2_c1/Cast:y:0'ssd_head/model/conv2_c1/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
.ssd_head/model/conv2_c1/BiasAdd/ReadVariableOpReadVariableOp7ssd_head_model_conv2_c1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
$ssd_head/model/conv2_c1/BiasAdd/CastCast6ssd_head/model/conv2_c1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
ssd_head/model/conv2_c1/BiasAddBiasAdd'ssd_head/model/conv2_c1/Conv2D:output:0(ssd_head/model/conv2_c1/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:������������
ssd_head/model/conv2_c1/ReluRelu(ssd_head/model/conv2_c1/BiasAdd:output:0*
T0*1
_output_shapes
:������������
#ssd_head/model/bn_c1/ReadVariableOpReadVariableOp,ssd_head_model_bn_c1_readvariableop_resource*
_output_shapes
:*
dtype0�
%ssd_head/model/bn_c1/ReadVariableOp_1ReadVariableOp.ssd_head_model_bn_c1_readvariableop_1_resource*
_output_shapes
:*
dtype0�
4ssd_head/model/bn_c1/FusedBatchNormV3/ReadVariableOpReadVariableOp=ssd_head_model_bn_c1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
6ssd_head/model/bn_c1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp?ssd_head_model_bn_c1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
%ssd_head/model/bn_c1/FusedBatchNormV3FusedBatchNormV3*ssd_head/model/conv2_c1/Relu:activations:0+ssd_head/model/bn_c1/ReadVariableOp:value:0-ssd_head/model/bn_c1/ReadVariableOp_1:value:0<ssd_head/model/bn_c1/FusedBatchNormV3/ReadVariableOp:value:0>ssd_head/model/bn_c1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:�����������:::::*
epsilon%o�:*
is_training( �
$ssd_head/model/max_pooling2d/MaxPoolMaxPool)ssd_head/model/bn_c1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@@*
ksize
*
paddingSAME*
strides
�
-ssd_head/model/conv2_c2/Conv2D/ReadVariableOpReadVariableOp6ssd_head_model_conv2_c2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
#ssd_head/model/conv2_c2/Conv2D/CastCast5ssd_head/model/conv2_c2/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
ssd_head/model/conv2_c2/Conv2DConv2D-ssd_head/model/max_pooling2d/MaxPool:output:0'ssd_head/model/conv2_c2/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
�
.ssd_head/model/conv2_c2/BiasAdd/ReadVariableOpReadVariableOp7ssd_head_model_conv2_c2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
$ssd_head/model/conv2_c2/BiasAdd/CastCast6ssd_head/model/conv2_c2/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
ssd_head/model/conv2_c2/BiasAddBiasAdd'ssd_head/model/conv2_c2/Conv2D:output:0(ssd_head/model/conv2_c2/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������@@�
ssd_head/model/conv2_c2/ReluRelu(ssd_head/model/conv2_c2/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@�
#ssd_head/model/bn_c2/ReadVariableOpReadVariableOp,ssd_head_model_bn_c2_readvariableop_resource*
_output_shapes
:*
dtype0�
%ssd_head/model/bn_c2/ReadVariableOp_1ReadVariableOp.ssd_head_model_bn_c2_readvariableop_1_resource*
_output_shapes
:*
dtype0�
4ssd_head/model/bn_c2/FusedBatchNormV3/ReadVariableOpReadVariableOp=ssd_head_model_bn_c2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
6ssd_head/model/bn_c2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp?ssd_head_model_bn_c2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
%ssd_head/model/bn_c2/FusedBatchNormV3FusedBatchNormV3*ssd_head/model/conv2_c2/Relu:activations:0+ssd_head/model/bn_c2/ReadVariableOp:value:0-ssd_head/model/bn_c2/ReadVariableOp_1:value:0<ssd_head/model/bn_c2/FusedBatchNormV3/ReadVariableOp:value:0>ssd_head/model/bn_c2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@@:::::*
epsilon%o�:*
is_training( �
&ssd_head/model/max_pooling2d_1/MaxPoolMaxPool)ssd_head/model/bn_c2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������  *
ksize
*
paddingSAME*
strides
�
-ssd_head/model/conv2_c3/Conv2D/ReadVariableOpReadVariableOp6ssd_head_model_conv2_c3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
#ssd_head/model/conv2_c3/Conv2D/CastCast5ssd_head/model/conv2_c3/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
ssd_head/model/conv2_c3/Conv2DConv2D/ssd_head/model/max_pooling2d_1/MaxPool:output:0'ssd_head/model/conv2_c3/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
�
.ssd_head/model/conv2_c3/BiasAdd/ReadVariableOpReadVariableOp7ssd_head_model_conv2_c3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
$ssd_head/model/conv2_c3/BiasAdd/CastCast6ssd_head/model/conv2_c3/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: �
ssd_head/model/conv2_c3/BiasAddBiasAdd'ssd_head/model/conv2_c3/Conv2D:output:0(ssd_head/model/conv2_c3/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������   �
ssd_head/model/conv2_c3/ReluRelu(ssd_head/model/conv2_c3/BiasAdd:output:0*
T0*/
_output_shapes
:���������   �
#ssd_head/model/bn_c3/ReadVariableOpReadVariableOp,ssd_head_model_bn_c3_readvariableop_resource*
_output_shapes
: *
dtype0�
%ssd_head/model/bn_c3/ReadVariableOp_1ReadVariableOp.ssd_head_model_bn_c3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
4ssd_head/model/bn_c3/FusedBatchNormV3/ReadVariableOpReadVariableOp=ssd_head_model_bn_c3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
6ssd_head/model/bn_c3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp?ssd_head_model_bn_c3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
%ssd_head/model/bn_c3/FusedBatchNormV3FusedBatchNormV3*ssd_head/model/conv2_c3/Relu:activations:0+ssd_head/model/bn_c3/ReadVariableOp:value:0-ssd_head/model/bn_c3/ReadVariableOp_1:value:0<ssd_head/model/bn_c3/FusedBatchNormV3/ReadVariableOp:value:0>ssd_head/model/bn_c3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
is_training( �
&ssd_head/model/max_pooling2d_2/MaxPoolMaxPool)ssd_head/model/bn_c3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:��������� *
ksize
*
paddingSAME*
strides
�
-ssd_head/model/conv2_c4/Conv2D/ReadVariableOpReadVariableOp6ssd_head_model_conv2_c4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
#ssd_head/model/conv2_c4/Conv2D/CastCast5ssd_head/model/conv2_c4/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:  �
ssd_head/model/conv2_c4/Conv2DConv2D/ssd_head/model/max_pooling2d_2/MaxPool:output:0'ssd_head/model/conv2_c4/Conv2D/Cast:y:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
.ssd_head/model/conv2_c4/BiasAdd/ReadVariableOpReadVariableOp7ssd_head_model_conv2_c4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
$ssd_head/model/conv2_c4/BiasAdd/CastCast6ssd_head/model/conv2_c4/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: �
ssd_head/model/conv2_c4/BiasAddBiasAdd'ssd_head/model/conv2_c4/Conv2D:output:0(ssd_head/model/conv2_c4/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:��������� �
ssd_head/model/conv2_c4/ReluRelu(ssd_head/model/conv2_c4/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
#ssd_head/model/bn_c4/ReadVariableOpReadVariableOp,ssd_head_model_bn_c4_readvariableop_resource*
_output_shapes
: *
dtype0�
%ssd_head/model/bn_c4/ReadVariableOp_1ReadVariableOp.ssd_head_model_bn_c4_readvariableop_1_resource*
_output_shapes
: *
dtype0�
4ssd_head/model/bn_c4/FusedBatchNormV3/ReadVariableOpReadVariableOp=ssd_head_model_bn_c4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
6ssd_head/model/bn_c4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp?ssd_head_model_bn_c4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
%ssd_head/model/bn_c4/FusedBatchNormV3FusedBatchNormV3*ssd_head/model/conv2_c4/Relu:activations:0+ssd_head/model/bn_c4/ReadVariableOp:value:0-ssd_head/model/bn_c4/ReadVariableOp_1:value:0<ssd_head/model/bn_c4/FusedBatchNormV3/ReadVariableOp:value:0>ssd_head/model/bn_c4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
is_training( �
&ssd_head/model/max_pooling2d_3/MaxPoolMaxPool)ssd_head/model/bn_c4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:��������� *
ksize
*
paddingSAME*
strides
�
-ssd_head/model/conv2_c5/Conv2D/ReadVariableOpReadVariableOp6ssd_head_model_conv2_c5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
#ssd_head/model/conv2_c5/Conv2D/CastCast5ssd_head/model/conv2_c5/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
ssd_head/model/conv2_c5/Conv2DConv2D/ssd_head/model/max_pooling2d_3/MaxPool:output:0'ssd_head/model/conv2_c5/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
.ssd_head/model/conv2_c5/BiasAdd/ReadVariableOpReadVariableOp7ssd_head_model_conv2_c5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
$ssd_head/model/conv2_c5/BiasAdd/CastCast6ssd_head/model/conv2_c5/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
ssd_head/model/conv2_c5/BiasAddBiasAdd'ssd_head/model/conv2_c5/Conv2D:output:0(ssd_head/model/conv2_c5/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:����������
ssd_head/model/conv2_c5/ReluRelu(ssd_head/model/conv2_c5/BiasAdd:output:0*
T0*/
_output_shapes
:����������
#ssd_head/off3/Conv2D/ReadVariableOpReadVariableOp,ssd_head_off3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
ssd_head/off3/Conv2D/CastCast+ssd_head/off3/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
ssd_head/off3/Conv2DConv2D*ssd_head/model/conv2_c5/Relu:activations:0ssd_head/off3/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
$ssd_head/off3/BiasAdd/ReadVariableOpReadVariableOp-ssd_head_off3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
ssd_head/off3/BiasAdd/CastCast,ssd_head/off3/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
ssd_head/off3/BiasAddBiasAddssd_head/off3/Conv2D:output:0ssd_head/off3/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:����������
#ssd_head/off2/Conv2D/ReadVariableOpReadVariableOp,ssd_head_off2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
ssd_head/off2/Conv2D/CastCast+ssd_head/off2/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
ssd_head/off2/Conv2DConv2D)ssd_head/model/bn_c4/FusedBatchNormV3:y:0ssd_head/off2/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
$ssd_head/off2/BiasAdd/ReadVariableOpReadVariableOp-ssd_head_off2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
ssd_head/off2/BiasAdd/CastCast,ssd_head/off2/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
ssd_head/off2/BiasAddBiasAddssd_head/off2/Conv2D:output:0ssd_head/off2/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:����������
#ssd_head/off1/Conv2D/ReadVariableOpReadVariableOp,ssd_head_off1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
ssd_head/off1/Conv2D/CastCast+ssd_head/off1/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
ssd_head/off1/Conv2DConv2D)ssd_head/model/bn_c3/FusedBatchNormV3:y:0ssd_head/off1/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
�
$ssd_head/off1/BiasAdd/ReadVariableOpReadVariableOp-ssd_head_off1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
ssd_head/off1/BiasAdd/CastCast,ssd_head/off1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
ssd_head/off1/BiasAddBiasAddssd_head/off1/Conv2D:output:0ssd_head/off1/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������  �
#ssd_head/cls3/Conv2D/ReadVariableOpReadVariableOp,ssd_head_cls3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
ssd_head/cls3/Conv2D/CastCast+ssd_head/cls3/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
ssd_head/cls3/Conv2DConv2D*ssd_head/model/conv2_c5/Relu:activations:0ssd_head/cls3/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
$ssd_head/cls3/BiasAdd/ReadVariableOpReadVariableOp-ssd_head_cls3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
ssd_head/cls3/BiasAdd/CastCast,ssd_head/cls3/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
ssd_head/cls3/BiasAddBiasAddssd_head/cls3/Conv2D:output:0ssd_head/cls3/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:����������
#ssd_head/cls2/Conv2D/ReadVariableOpReadVariableOp,ssd_head_cls2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
ssd_head/cls2/Conv2D/CastCast+ssd_head/cls2/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
ssd_head/cls2/Conv2DConv2D)ssd_head/model/bn_c4/FusedBatchNormV3:y:0ssd_head/cls2/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
$ssd_head/cls2/BiasAdd/ReadVariableOpReadVariableOp-ssd_head_cls2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
ssd_head/cls2/BiasAdd/CastCast,ssd_head/cls2/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
ssd_head/cls2/BiasAddBiasAddssd_head/cls2/Conv2D:output:0ssd_head/cls2/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:����������
#ssd_head/cls1/Conv2D/ReadVariableOpReadVariableOp,ssd_head_cls1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
ssd_head/cls1/Conv2D/CastCast+ssd_head/cls1/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
ssd_head/cls1/Conv2DConv2D)ssd_head/model/bn_c3/FusedBatchNormV3:y:0ssd_head/cls1/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
�
$ssd_head/cls1/BiasAdd/ReadVariableOpReadVariableOp-ssd_head_cls1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
ssd_head/cls1/BiasAdd/CastCast,ssd_head/cls1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
ssd_head/cls1/BiasAddBiasAddssd_head/cls1/Conv2D:output:0ssd_head/cls1/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������  e
ssd_head/off_res3/ShapeShapessd_head/off3/BiasAdd:output:0*
T0*
_output_shapes
:o
%ssd_head/off_res3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'ssd_head/off_res3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'ssd_head/off_res3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
ssd_head/off_res3/strided_sliceStridedSlice ssd_head/off_res3/Shape:output:0.ssd_head/off_res3/strided_slice/stack:output:00ssd_head/off_res3/strided_slice/stack_1:output:00ssd_head/off_res3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!ssd_head/off_res3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������c
!ssd_head/off_res3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
ssd_head/off_res3/Reshape/shapePack(ssd_head/off_res3/strided_slice:output:0*ssd_head/off_res3/Reshape/shape/1:output:0*ssd_head/off_res3/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
ssd_head/off_res3/ReshapeReshapessd_head/off3/BiasAdd:output:0(ssd_head/off_res3/Reshape/shape:output:0*
T0*,
_output_shapes
:����������e
ssd_head/off_res2/ShapeShapessd_head/off2/BiasAdd:output:0*
T0*
_output_shapes
:o
%ssd_head/off_res2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'ssd_head/off_res2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'ssd_head/off_res2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
ssd_head/off_res2/strided_sliceStridedSlice ssd_head/off_res2/Shape:output:0.ssd_head/off_res2/strided_slice/stack:output:00ssd_head/off_res2/strided_slice/stack_1:output:00ssd_head/off_res2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!ssd_head/off_res2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������c
!ssd_head/off_res2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
ssd_head/off_res2/Reshape/shapePack(ssd_head/off_res2/strided_slice:output:0*ssd_head/off_res2/Reshape/shape/1:output:0*ssd_head/off_res2/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
ssd_head/off_res2/ReshapeReshapessd_head/off2/BiasAdd:output:0(ssd_head/off_res2/Reshape/shape:output:0*
T0*,
_output_shapes
:����������e
ssd_head/off_res1/ShapeShapessd_head/off1/BiasAdd:output:0*
T0*
_output_shapes
:o
%ssd_head/off_res1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'ssd_head/off_res1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'ssd_head/off_res1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
ssd_head/off_res1/strided_sliceStridedSlice ssd_head/off_res1/Shape:output:0.ssd_head/off_res1/strided_slice/stack:output:00ssd_head/off_res1/strided_slice/stack_1:output:00ssd_head/off_res1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!ssd_head/off_res1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������c
!ssd_head/off_res1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
ssd_head/off_res1/Reshape/shapePack(ssd_head/off_res1/strided_slice:output:0*ssd_head/off_res1/Reshape/shape/1:output:0*ssd_head/off_res1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
ssd_head/off_res1/ReshapeReshapessd_head/off1/BiasAdd:output:0(ssd_head/off_res1/Reshape/shape:output:0*
T0*,
_output_shapes
:���������� e
ssd_head/cls_res3/ShapeShapessd_head/cls3/BiasAdd:output:0*
T0*
_output_shapes
:o
%ssd_head/cls_res3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'ssd_head/cls_res3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'ssd_head/cls_res3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
ssd_head/cls_res3/strided_sliceStridedSlice ssd_head/cls_res3/Shape:output:0.ssd_head/cls_res3/strided_slice/stack:output:00ssd_head/cls_res3/strided_slice/stack_1:output:00ssd_head/cls_res3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!ssd_head/cls_res3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������c
!ssd_head/cls_res3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
ssd_head/cls_res3/Reshape/shapePack(ssd_head/cls_res3/strided_slice:output:0*ssd_head/cls_res3/Reshape/shape/1:output:0*ssd_head/cls_res3/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
ssd_head/cls_res3/ReshapeReshapessd_head/cls3/BiasAdd:output:0(ssd_head/cls_res3/Reshape/shape:output:0*
T0*,
_output_shapes
:����������e
ssd_head/cls_res2/ShapeShapessd_head/cls2/BiasAdd:output:0*
T0*
_output_shapes
:o
%ssd_head/cls_res2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'ssd_head/cls_res2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'ssd_head/cls_res2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
ssd_head/cls_res2/strided_sliceStridedSlice ssd_head/cls_res2/Shape:output:0.ssd_head/cls_res2/strided_slice/stack:output:00ssd_head/cls_res2/strided_slice/stack_1:output:00ssd_head/cls_res2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!ssd_head/cls_res2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������c
!ssd_head/cls_res2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
ssd_head/cls_res2/Reshape/shapePack(ssd_head/cls_res2/strided_slice:output:0*ssd_head/cls_res2/Reshape/shape/1:output:0*ssd_head/cls_res2/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
ssd_head/cls_res2/ReshapeReshapessd_head/cls2/BiasAdd:output:0(ssd_head/cls_res2/Reshape/shape:output:0*
T0*,
_output_shapes
:����������e
ssd_head/cls_res1/ShapeShapessd_head/cls1/BiasAdd:output:0*
T0*
_output_shapes
:o
%ssd_head/cls_res1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'ssd_head/cls_res1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'ssd_head/cls_res1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
ssd_head/cls_res1/strided_sliceStridedSlice ssd_head/cls_res1/Shape:output:0.ssd_head/cls_res1/strided_slice/stack:output:00ssd_head/cls_res1/strided_slice/stack_1:output:00ssd_head/cls_res1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!ssd_head/cls_res1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������c
!ssd_head/cls_res1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
ssd_head/cls_res1/Reshape/shapePack(ssd_head/cls_res1/strided_slice:output:0*ssd_head/cls_res1/Reshape/shape/1:output:0*ssd_head/cls_res1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
ssd_head/cls_res1/ReshapeReshapessd_head/cls1/BiasAdd:output:0(ssd_head/cls_res1/Reshape/shape:output:0*
T0*,
_output_shapes
:���������� _
ssd_head/off_cat1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
ssd_head/off_cat1/concatConcatV2"ssd_head/off_res1/Reshape:output:0"ssd_head/off_res1/Reshape:output:0&ssd_head/off_cat1/concat/axis:output:0*
N*
T0*,
_output_shapes
:���������� _
ssd_head/off_cat2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
ssd_head/off_cat2/concatConcatV2"ssd_head/off_res2/Reshape:output:0"ssd_head/off_res2/Reshape:output:0&ssd_head/off_cat2/concat/axis:output:0*
N*
T0*,
_output_shapes
:����������_
ssd_head/off_cat3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
ssd_head/off_cat3/concatConcatV2"ssd_head/off_res3/Reshape:output:0"ssd_head/off_res3/Reshape:output:0&ssd_head/off_cat3/concat/axis:output:0*
N*
T0*,
_output_shapes
:����������
ssd_head/cls_out1/SoftmaxSoftmax"ssd_head/cls_res1/Reshape:output:0*
T0*,
_output_shapes
:���������� 
ssd_head/cls_out2/SoftmaxSoftmax"ssd_head/cls_res2/Reshape:output:0*
T0*,
_output_shapes
:����������
ssd_head/cls_out3/SoftmaxSoftmax"ssd_head/cls_res3/Reshape:output:0*
T0*,
_output_shapes
:����������^
ssd_head/offsets/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
ssd_head/offsets/concatConcatV2!ssd_head/off_cat1/concat:output:0!ssd_head/off_cat2/concat:output:0!ssd_head/off_cat3/concat:output:0%ssd_head/offsets/concat/axis:output:0*
N*
T0*,
_output_shapes
:����������*^
ssd_head/classes/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
ssd_head/classes/concatConcatV2#ssd_head/cls_out1/Softmax:softmax:0#ssd_head/cls_out2/Softmax:softmax:0#ssd_head/cls_out3/Softmax:softmax:0%ssd_head/classes/concat/axis:output:0*
N*
T0*,
_output_shapes
:����������*t
IdentityIdentity ssd_head/classes/concat:output:0^NoOp*
T0*,
_output_shapes
:����������*v

Identity_1Identity ssd_head/offsets/concat:output:0^NoOp*
T0*,
_output_shapes
:����������*�
NoOpNoOp%^ssd_head/cls1/BiasAdd/ReadVariableOp$^ssd_head/cls1/Conv2D/ReadVariableOp%^ssd_head/cls2/BiasAdd/ReadVariableOp$^ssd_head/cls2/Conv2D/ReadVariableOp%^ssd_head/cls3/BiasAdd/ReadVariableOp$^ssd_head/cls3/Conv2D/ReadVariableOp5^ssd_head/model/bn_c1/FusedBatchNormV3/ReadVariableOp7^ssd_head/model/bn_c1/FusedBatchNormV3/ReadVariableOp_1$^ssd_head/model/bn_c1/ReadVariableOp&^ssd_head/model/bn_c1/ReadVariableOp_15^ssd_head/model/bn_c2/FusedBatchNormV3/ReadVariableOp7^ssd_head/model/bn_c2/FusedBatchNormV3/ReadVariableOp_1$^ssd_head/model/bn_c2/ReadVariableOp&^ssd_head/model/bn_c2/ReadVariableOp_15^ssd_head/model/bn_c3/FusedBatchNormV3/ReadVariableOp7^ssd_head/model/bn_c3/FusedBatchNormV3/ReadVariableOp_1$^ssd_head/model/bn_c3/ReadVariableOp&^ssd_head/model/bn_c3/ReadVariableOp_15^ssd_head/model/bn_c4/FusedBatchNormV3/ReadVariableOp7^ssd_head/model/bn_c4/FusedBatchNormV3/ReadVariableOp_1$^ssd_head/model/bn_c4/ReadVariableOp&^ssd_head/model/bn_c4/ReadVariableOp_1/^ssd_head/model/conv2_c1/BiasAdd/ReadVariableOp.^ssd_head/model/conv2_c1/Conv2D/ReadVariableOp/^ssd_head/model/conv2_c2/BiasAdd/ReadVariableOp.^ssd_head/model/conv2_c2/Conv2D/ReadVariableOp/^ssd_head/model/conv2_c3/BiasAdd/ReadVariableOp.^ssd_head/model/conv2_c3/Conv2D/ReadVariableOp/^ssd_head/model/conv2_c4/BiasAdd/ReadVariableOp.^ssd_head/model/conv2_c4/Conv2D/ReadVariableOp/^ssd_head/model/conv2_c5/BiasAdd/ReadVariableOp.^ssd_head/model/conv2_c5/Conv2D/ReadVariableOp%^ssd_head/off1/BiasAdd/ReadVariableOp$^ssd_head/off1/Conv2D/ReadVariableOp%^ssd_head/off2/BiasAdd/ReadVariableOp$^ssd_head/off2/Conv2D/ReadVariableOp%^ssd_head/off3/BiasAdd/ReadVariableOp$^ssd_head/off3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$ssd_head/cls1/BiasAdd/ReadVariableOp$ssd_head/cls1/BiasAdd/ReadVariableOp2J
#ssd_head/cls1/Conv2D/ReadVariableOp#ssd_head/cls1/Conv2D/ReadVariableOp2L
$ssd_head/cls2/BiasAdd/ReadVariableOp$ssd_head/cls2/BiasAdd/ReadVariableOp2J
#ssd_head/cls2/Conv2D/ReadVariableOp#ssd_head/cls2/Conv2D/ReadVariableOp2L
$ssd_head/cls3/BiasAdd/ReadVariableOp$ssd_head/cls3/BiasAdd/ReadVariableOp2J
#ssd_head/cls3/Conv2D/ReadVariableOp#ssd_head/cls3/Conv2D/ReadVariableOp2l
4ssd_head/model/bn_c1/FusedBatchNormV3/ReadVariableOp4ssd_head/model/bn_c1/FusedBatchNormV3/ReadVariableOp2p
6ssd_head/model/bn_c1/FusedBatchNormV3/ReadVariableOp_16ssd_head/model/bn_c1/FusedBatchNormV3/ReadVariableOp_12J
#ssd_head/model/bn_c1/ReadVariableOp#ssd_head/model/bn_c1/ReadVariableOp2N
%ssd_head/model/bn_c1/ReadVariableOp_1%ssd_head/model/bn_c1/ReadVariableOp_12l
4ssd_head/model/bn_c2/FusedBatchNormV3/ReadVariableOp4ssd_head/model/bn_c2/FusedBatchNormV3/ReadVariableOp2p
6ssd_head/model/bn_c2/FusedBatchNormV3/ReadVariableOp_16ssd_head/model/bn_c2/FusedBatchNormV3/ReadVariableOp_12J
#ssd_head/model/bn_c2/ReadVariableOp#ssd_head/model/bn_c2/ReadVariableOp2N
%ssd_head/model/bn_c2/ReadVariableOp_1%ssd_head/model/bn_c2/ReadVariableOp_12l
4ssd_head/model/bn_c3/FusedBatchNormV3/ReadVariableOp4ssd_head/model/bn_c3/FusedBatchNormV3/ReadVariableOp2p
6ssd_head/model/bn_c3/FusedBatchNormV3/ReadVariableOp_16ssd_head/model/bn_c3/FusedBatchNormV3/ReadVariableOp_12J
#ssd_head/model/bn_c3/ReadVariableOp#ssd_head/model/bn_c3/ReadVariableOp2N
%ssd_head/model/bn_c3/ReadVariableOp_1%ssd_head/model/bn_c3/ReadVariableOp_12l
4ssd_head/model/bn_c4/FusedBatchNormV3/ReadVariableOp4ssd_head/model/bn_c4/FusedBatchNormV3/ReadVariableOp2p
6ssd_head/model/bn_c4/FusedBatchNormV3/ReadVariableOp_16ssd_head/model/bn_c4/FusedBatchNormV3/ReadVariableOp_12J
#ssd_head/model/bn_c4/ReadVariableOp#ssd_head/model/bn_c4/ReadVariableOp2N
%ssd_head/model/bn_c4/ReadVariableOp_1%ssd_head/model/bn_c4/ReadVariableOp_12`
.ssd_head/model/conv2_c1/BiasAdd/ReadVariableOp.ssd_head/model/conv2_c1/BiasAdd/ReadVariableOp2^
-ssd_head/model/conv2_c1/Conv2D/ReadVariableOp-ssd_head/model/conv2_c1/Conv2D/ReadVariableOp2`
.ssd_head/model/conv2_c2/BiasAdd/ReadVariableOp.ssd_head/model/conv2_c2/BiasAdd/ReadVariableOp2^
-ssd_head/model/conv2_c2/Conv2D/ReadVariableOp-ssd_head/model/conv2_c2/Conv2D/ReadVariableOp2`
.ssd_head/model/conv2_c3/BiasAdd/ReadVariableOp.ssd_head/model/conv2_c3/BiasAdd/ReadVariableOp2^
-ssd_head/model/conv2_c3/Conv2D/ReadVariableOp-ssd_head/model/conv2_c3/Conv2D/ReadVariableOp2`
.ssd_head/model/conv2_c4/BiasAdd/ReadVariableOp.ssd_head/model/conv2_c4/BiasAdd/ReadVariableOp2^
-ssd_head/model/conv2_c4/Conv2D/ReadVariableOp-ssd_head/model/conv2_c4/Conv2D/ReadVariableOp2`
.ssd_head/model/conv2_c5/BiasAdd/ReadVariableOp.ssd_head/model/conv2_c5/BiasAdd/ReadVariableOp2^
-ssd_head/model/conv2_c5/Conv2D/ReadVariableOp-ssd_head/model/conv2_c5/Conv2D/ReadVariableOp2L
$ssd_head/off1/BiasAdd/ReadVariableOp$ssd_head/off1/BiasAdd/ReadVariableOp2J
#ssd_head/off1/Conv2D/ReadVariableOp#ssd_head/off1/Conv2D/ReadVariableOp2L
$ssd_head/off2/BiasAdd/ReadVariableOp$ssd_head/off2/BiasAdd/ReadVariableOp2J
#ssd_head/off2/Conv2D/ReadVariableOp#ssd_head/off2/Conv2D/ReadVariableOp2L
$ssd_head/off3/BiasAdd/ReadVariableOp$ssd_head/off3/BiasAdd/ReadVariableOp2J
#ssd_head/off3/Conv2D/ReadVariableOp#ssd_head/off3/Conv2D/ReadVariableOp:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
e
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1295

inputs
identity�
MaxPoolMaxPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�@
�	
?__inference_model_layer_call_and_return_conditional_losses_1688

inputs'
conv2_c1_1620:
conv2_c1_1622:

bn_c1_1625:

bn_c1_1627:

bn_c1_1629:

bn_c1_1631:'
conv2_c2_1635:
conv2_c2_1637:

bn_c2_1640:

bn_c2_1642:

bn_c2_1644:

bn_c2_1646:'
conv2_c3_1650: 
conv2_c3_1652: 

bn_c3_1655: 

bn_c3_1657: 

bn_c3_1659: 

bn_c3_1661: '
conv2_c4_1665:  
conv2_c4_1667: 

bn_c4_1670: 

bn_c4_1672: 

bn_c4_1674: 

bn_c4_1676: '
conv2_c5_1680: 
conv2_c5_1682:
identity

identity_1

identity_2��bn_c1/StatefulPartitionedCall�bn_c2/StatefulPartitionedCall�bn_c3/StatefulPartitionedCall�bn_c4/StatefulPartitionedCall� conv2_c1/StatefulPartitionedCall� conv2_c2/StatefulPartitionedCall� conv2_c3/StatefulPartitionedCall� conv2_c4/StatefulPartitionedCall� conv2_c5/StatefulPartitionedCallh
conv2_c1/CastCastinputs*

DstT0*

SrcT0*1
_output_shapes
:������������
 conv2_c1/StatefulPartitionedCallStatefulPartitionedCallconv2_c1/Cast:y:0conv2_c1_1620conv2_c1_1622*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2_c1_layer_call_and_return_conditional_losses_1319�
bn_c1/StatefulPartitionedCallStatefulPartitionedCall)conv2_c1/StatefulPartitionedCall:output:0
bn_c1_1625
bn_c1_1627
bn_c1_1629
bn_c1_1631*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_bn_c1_layer_call_and_return_conditional_losses_1047�
max_pooling2d/PartitionedCallPartitionedCall&bn_c1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1067�
 conv2_c2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2_c2_1635conv2_c2_1637*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2_c2_layer_call_and_return_conditional_losses_1348�
bn_c2/StatefulPartitionedCallStatefulPartitionedCall)conv2_c2/StatefulPartitionedCall:output:0
bn_c2_1640
bn_c2_1642
bn_c2_1644
bn_c2_1646*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_bn_c2_layer_call_and_return_conditional_losses_1123�
max_pooling2d_1/PartitionedCallPartitionedCall&bn_c2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1143�
 conv2_c3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2_c3_1650conv2_c3_1652*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2_c3_layer_call_and_return_conditional_losses_1377�
bn_c3/StatefulPartitionedCallStatefulPartitionedCall)conv2_c3/StatefulPartitionedCall:output:0
bn_c3_1655
bn_c3_1657
bn_c3_1659
bn_c3_1661*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_bn_c3_layer_call_and_return_conditional_losses_1199�
max_pooling2d_2/PartitionedCallPartitionedCall&bn_c3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1219�
 conv2_c4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2_c4_1665conv2_c4_1667*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2_c4_layer_call_and_return_conditional_losses_1406�
bn_c4/StatefulPartitionedCallStatefulPartitionedCall)conv2_c4/StatefulPartitionedCall:output:0
bn_c4_1670
bn_c4_1672
bn_c4_1674
bn_c4_1676*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_bn_c4_layer_call_and_return_conditional_losses_1275�
max_pooling2d_3/PartitionedCallPartitionedCall&bn_c4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1295�
 conv2_c5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2_c5_1680conv2_c5_1682*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2_c5_layer_call_and_return_conditional_losses_1435}
IdentityIdentity&bn_c3/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������   

Identity_1Identity&bn_c4/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� �

Identity_2Identity)conv2_c5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^bn_c1/StatefulPartitionedCall^bn_c2/StatefulPartitionedCall^bn_c3/StatefulPartitionedCall^bn_c4/StatefulPartitionedCall!^conv2_c1/StatefulPartitionedCall!^conv2_c2/StatefulPartitionedCall!^conv2_c3/StatefulPartitionedCall!^conv2_c4/StatefulPartitionedCall!^conv2_c5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : 2>
bn_c1/StatefulPartitionedCallbn_c1/StatefulPartitionedCall2>
bn_c2/StatefulPartitionedCallbn_c2/StatefulPartitionedCall2>
bn_c3/StatefulPartitionedCallbn_c3/StatefulPartitionedCall2>
bn_c4/StatefulPartitionedCallbn_c4/StatefulPartitionedCall2D
 conv2_c1/StatefulPartitionedCall conv2_c1/StatefulPartitionedCall2D
 conv2_c2/StatefulPartitionedCall conv2_c2/StatefulPartitionedCall2D
 conv2_c3/StatefulPartitionedCall conv2_c3/StatefulPartitionedCall2D
 conv2_c4/StatefulPartitionedCall conv2_c4/StatefulPartitionedCall2D
 conv2_c5/StatefulPartitionedCall conv2_c5/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
^
B__inference_cls_out2_layer_call_and_return_conditional_losses_4378

inputs
identityQ
SoftmaxSoftmaxinputs*
T0*,
_output_shapes
:����������^
IdentityIdentitySoftmax:softmax:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�U
�
B__inference_ssd_head_layer_call_and_return_conditional_losses_2971
input_1$

model_2870:

model_2872:

model_2874:

model_2876:

model_2878:

model_2880:$

model_2882:

model_2884:

model_2886:

model_2888:

model_2890:

model_2892:$

model_2894: 

model_2896: 

model_2898: 

model_2900: 

model_2902: 

model_2904: $

model_2906:  

model_2908: 

model_2910: 

model_2912: 

model_2914: 

model_2916: $

model_2918: 

model_2920:#
	off3_2925:
	off3_2927:#
	off2_2930: 
	off2_2932:#
	off1_2935: 
	off1_2937:#
	cls3_2940:
	cls3_2942:#
	cls2_2945: 
	cls2_2947:#
	cls1_2950: 
	cls1_2952:
identity

identity_1��cls1/StatefulPartitionedCall�cls2/StatefulPartitionedCall�cls3/StatefulPartitionedCall�model/StatefulPartitionedCall�off1/StatefulPartitionedCall�off2/StatefulPartitionedCall�off3/StatefulPartitionedCall�
model/StatefulPartitionedCallStatefulPartitionedCallinput_1
model_2870
model_2872
model_2874
model_2876
model_2878
model_2880
model_2882
model_2884
model_2886
model_2888
model_2890
model_2892
model_2894
model_2896
model_2898
model_2900
model_2902
model_2904
model_2906
model_2908
model_2910
model_2912
model_2914
model_2916
model_2918
model_2920*&
Tin
2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:���������   :��������� :���������*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_1444�
off3/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:2	off3_2925	off3_2927*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_off3_layer_call_and_return_conditional_losses_2026�
off2/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:1	off2_2930	off2_2932*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_off2_layer_call_and_return_conditional_losses_2044�
off1/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0	off1_2935	off1_2937*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_off1_layer_call_and_return_conditional_losses_2062�
cls3/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:2	cls3_2940	cls3_2942*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_cls3_layer_call_and_return_conditional_losses_2080�
cls2/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:1	cls2_2945	cls2_2947*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_cls2_layer_call_and_return_conditional_losses_2098�
cls1/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0	cls1_2950	cls1_2952*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_cls1_layer_call_and_return_conditional_losses_2116�
off_res3/PartitionedCallPartitionedCall%off3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_off_res3_layer_call_and_return_conditional_losses_2135�
off_res2/PartitionedCallPartitionedCall%off2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_off_res2_layer_call_and_return_conditional_losses_2150�
off_res1/PartitionedCallPartitionedCall%off1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_off_res1_layer_call_and_return_conditional_losses_2165�
cls_res3/PartitionedCallPartitionedCall%cls3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_cls_res3_layer_call_and_return_conditional_losses_2180�
cls_res2/PartitionedCallPartitionedCall%cls2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_cls_res2_layer_call_and_return_conditional_losses_2195�
cls_res1/PartitionedCallPartitionedCall%cls1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_cls_res1_layer_call_and_return_conditional_losses_2210�
off_cat1/PartitionedCallPartitionedCall!off_res1/PartitionedCall:output:0!off_res1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_off_cat1_layer_call_and_return_conditional_losses_2219�
off_cat2/PartitionedCallPartitionedCall!off_res2/PartitionedCall:output:0!off_res2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_off_cat2_layer_call_and_return_conditional_losses_2228�
off_cat3/PartitionedCallPartitionedCall!off_res3/PartitionedCall:output:0!off_res3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_off_cat3_layer_call_and_return_conditional_losses_2237�
cls_out1/PartitionedCallPartitionedCall!cls_res1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_cls_out1_layer_call_and_return_conditional_losses_2244�
cls_out2/PartitionedCallPartitionedCall!cls_res2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_cls_out2_layer_call_and_return_conditional_losses_2251�
cls_out3/PartitionedCallPartitionedCall!cls_res3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_cls_out3_layer_call_and_return_conditional_losses_2258�
offsets/PartitionedCallPartitionedCall!off_cat1/PartitionedCall:output:0!off_cat2/PartitionedCall:output:0!off_cat3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_offsets_layer_call_and_return_conditional_losses_2268�
classes/PartitionedCallPartitionedCall!cls_out1/PartitionedCall:output:0!cls_out2/PartitionedCall:output:0!cls_out3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_classes_layer_call_and_return_conditional_losses_2278t
IdentityIdentity classes/PartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������*v

Identity_1Identity offsets/PartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������*�
NoOpNoOp^cls1/StatefulPartitionedCall^cls2/StatefulPartitionedCall^cls3/StatefulPartitionedCall^model/StatefulPartitionedCall^off1/StatefulPartitionedCall^off2/StatefulPartitionedCall^off3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
cls1/StatefulPartitionedCallcls1/StatefulPartitionedCall2<
cls2/StatefulPartitionedCallcls2/StatefulPartitionedCall2<
cls3/StatefulPartitionedCallcls3/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2<
off1/StatefulPartitionedCalloff1/StatefulPartitionedCall2<
off2/StatefulPartitionedCalloff2/StatefulPartitionedCall2<
off3/StatefulPartitionedCalloff3/StatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
�
'__inference_conv2_c3_layer_call_fn_4654

inputs!
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2_c3_layer_call_and_return_conditional_losses_1377w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�H
�
__inference__traced_save_4993
file_prefix*
&savev2_cls1_kernel_read_readvariableop(
$savev2_cls1_bias_read_readvariableop*
&savev2_cls2_kernel_read_readvariableop(
$savev2_cls2_bias_read_readvariableop*
&savev2_cls3_kernel_read_readvariableop(
$savev2_cls3_bias_read_readvariableop*
&savev2_off1_kernel_read_readvariableop(
$savev2_off1_bias_read_readvariableop*
&savev2_off2_kernel_read_readvariableop(
$savev2_off2_bias_read_readvariableop*
&savev2_off3_kernel_read_readvariableop(
$savev2_off3_bias_read_readvariableop.
*savev2_conv2_c1_kernel_read_readvariableop,
(savev2_conv2_c1_bias_read_readvariableop*
&savev2_bn_c1_gamma_read_readvariableop)
%savev2_bn_c1_beta_read_readvariableop0
,savev2_bn_c1_moving_mean_read_readvariableop4
0savev2_bn_c1_moving_variance_read_readvariableop.
*savev2_conv2_c2_kernel_read_readvariableop,
(savev2_conv2_c2_bias_read_readvariableop*
&savev2_bn_c2_gamma_read_readvariableop)
%savev2_bn_c2_beta_read_readvariableop0
,savev2_bn_c2_moving_mean_read_readvariableop4
0savev2_bn_c2_moving_variance_read_readvariableop.
*savev2_conv2_c3_kernel_read_readvariableop,
(savev2_conv2_c3_bias_read_readvariableop*
&savev2_bn_c3_gamma_read_readvariableop)
%savev2_bn_c3_beta_read_readvariableop0
,savev2_bn_c3_moving_mean_read_readvariableop4
0savev2_bn_c3_moving_variance_read_readvariableop.
*savev2_conv2_c4_kernel_read_readvariableop,
(savev2_conv2_c4_bias_read_readvariableop*
&savev2_bn_c4_gamma_read_readvariableop)
%savev2_bn_c4_beta_read_readvariableop0
,savev2_bn_c4_moving_mean_read_readvariableop4
0savev2_bn_c4_moving_variance_read_readvariableop.
*savev2_conv2_c5_kernel_read_readvariableop,
(savev2_conv2_c5_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*�
value�B�'B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*a
valueXBV'B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_cls1_kernel_read_readvariableop$savev2_cls1_bias_read_readvariableop&savev2_cls2_kernel_read_readvariableop$savev2_cls2_bias_read_readvariableop&savev2_cls3_kernel_read_readvariableop$savev2_cls3_bias_read_readvariableop&savev2_off1_kernel_read_readvariableop$savev2_off1_bias_read_readvariableop&savev2_off2_kernel_read_readvariableop$savev2_off2_bias_read_readvariableop&savev2_off3_kernel_read_readvariableop$savev2_off3_bias_read_readvariableop*savev2_conv2_c1_kernel_read_readvariableop(savev2_conv2_c1_bias_read_readvariableop&savev2_bn_c1_gamma_read_readvariableop%savev2_bn_c1_beta_read_readvariableop,savev2_bn_c1_moving_mean_read_readvariableop0savev2_bn_c1_moving_variance_read_readvariableop*savev2_conv2_c2_kernel_read_readvariableop(savev2_conv2_c2_bias_read_readvariableop&savev2_bn_c2_gamma_read_readvariableop%savev2_bn_c2_beta_read_readvariableop,savev2_bn_c2_moving_mean_read_readvariableop0savev2_bn_c2_moving_variance_read_readvariableop*savev2_conv2_c3_kernel_read_readvariableop(savev2_conv2_c3_bias_read_readvariableop&savev2_bn_c3_gamma_read_readvariableop%savev2_bn_c3_beta_read_readvariableop,savev2_bn_c3_moving_mean_read_readvariableop0savev2_bn_c3_moving_variance_read_readvariableop*savev2_conv2_c4_kernel_read_readvariableop(savev2_conv2_c4_bias_read_readvariableop&savev2_bn_c4_gamma_read_readvariableop%savev2_bn_c4_beta_read_readvariableop,savev2_bn_c4_moving_mean_read_readvariableop0savev2_bn_c4_moving_variance_read_readvariableop*savev2_conv2_c5_kernel_read_readvariableop(savev2_conv2_c5_bias_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *5
dtypes+
)2'�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: : :: :::: :: :::::::::::::::: : : : : : :  : : : : : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
::,	(
&
_output_shapes
: : 


_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  :  

_output_shapes
: : !

_output_shapes
: : "

_output_shapes
: : #

_output_shapes
: : $

_output_shapes
: :,%(
&
_output_shapes
: : &

_output_shapes
::'

_output_shapes
: 
�@
�	
?__inference_model_layer_call_and_return_conditional_losses_1444

inputs'
conv2_c1_1320:
conv2_c1_1322:

bn_c1_1325:

bn_c1_1327:

bn_c1_1329:

bn_c1_1331:'
conv2_c2_1349:
conv2_c2_1351:

bn_c2_1354:

bn_c2_1356:

bn_c2_1358:

bn_c2_1360:'
conv2_c3_1378: 
conv2_c3_1380: 

bn_c3_1383: 

bn_c3_1385: 

bn_c3_1387: 

bn_c3_1389: '
conv2_c4_1407:  
conv2_c4_1409: 

bn_c4_1412: 

bn_c4_1414: 

bn_c4_1416: 

bn_c4_1418: '
conv2_c5_1436: 
conv2_c5_1438:
identity

identity_1

identity_2��bn_c1/StatefulPartitionedCall�bn_c2/StatefulPartitionedCall�bn_c3/StatefulPartitionedCall�bn_c4/StatefulPartitionedCall� conv2_c1/StatefulPartitionedCall� conv2_c2/StatefulPartitionedCall� conv2_c3/StatefulPartitionedCall� conv2_c4/StatefulPartitionedCall� conv2_c5/StatefulPartitionedCallh
conv2_c1/CastCastinputs*

DstT0*

SrcT0*1
_output_shapes
:������������
 conv2_c1/StatefulPartitionedCallStatefulPartitionedCallconv2_c1/Cast:y:0conv2_c1_1320conv2_c1_1322*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2_c1_layer_call_and_return_conditional_losses_1319�
bn_c1/StatefulPartitionedCallStatefulPartitionedCall)conv2_c1/StatefulPartitionedCall:output:0
bn_c1_1325
bn_c1_1327
bn_c1_1329
bn_c1_1331*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_bn_c1_layer_call_and_return_conditional_losses_1016�
max_pooling2d/PartitionedCallPartitionedCall&bn_c1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1067�
 conv2_c2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2_c2_1349conv2_c2_1351*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2_c2_layer_call_and_return_conditional_losses_1348�
bn_c2/StatefulPartitionedCallStatefulPartitionedCall)conv2_c2/StatefulPartitionedCall:output:0
bn_c2_1354
bn_c2_1356
bn_c2_1358
bn_c2_1360*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_bn_c2_layer_call_and_return_conditional_losses_1092�
max_pooling2d_1/PartitionedCallPartitionedCall&bn_c2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1143�
 conv2_c3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2_c3_1378conv2_c3_1380*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2_c3_layer_call_and_return_conditional_losses_1377�
bn_c3/StatefulPartitionedCallStatefulPartitionedCall)conv2_c3/StatefulPartitionedCall:output:0
bn_c3_1383
bn_c3_1385
bn_c3_1387
bn_c3_1389*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_bn_c3_layer_call_and_return_conditional_losses_1168�
max_pooling2d_2/PartitionedCallPartitionedCall&bn_c3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1219�
 conv2_c4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2_c4_1407conv2_c4_1409*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2_c4_layer_call_and_return_conditional_losses_1406�
bn_c4/StatefulPartitionedCallStatefulPartitionedCall)conv2_c4/StatefulPartitionedCall:output:0
bn_c4_1412
bn_c4_1414
bn_c4_1416
bn_c4_1418*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_bn_c4_layer_call_and_return_conditional_losses_1244�
max_pooling2d_3/PartitionedCallPartitionedCall&bn_c4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1295�
 conv2_c5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2_c5_1436conv2_c5_1438*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2_c5_layer_call_and_return_conditional_losses_1435}
IdentityIdentity&bn_c3/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������   

Identity_1Identity&bn_c4/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� �

Identity_2Identity)conv2_c5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^bn_c1/StatefulPartitionedCall^bn_c2/StatefulPartitionedCall^bn_c3/StatefulPartitionedCall^bn_c4/StatefulPartitionedCall!^conv2_c1/StatefulPartitionedCall!^conv2_c2/StatefulPartitionedCall!^conv2_c3/StatefulPartitionedCall!^conv2_c4/StatefulPartitionedCall!^conv2_c5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : 2>
bn_c1/StatefulPartitionedCallbn_c1/StatefulPartitionedCall2>
bn_c2/StatefulPartitionedCallbn_c2/StatefulPartitionedCall2>
bn_c3/StatefulPartitionedCallbn_c3/StatefulPartitionedCall2>
bn_c4/StatefulPartitionedCallbn_c4/StatefulPartitionedCall2D
 conv2_c1/StatefulPartitionedCall conv2_c1/StatefulPartitionedCall2D
 conv2_c2/StatefulPartitionedCall conv2_c2/StatefulPartitionedCall2D
 conv2_c3/StatefulPartitionedCall conv2_c3/StatefulPartitionedCall2D
 conv2_c4/StatefulPartitionedCall conv2_c4/StatefulPartitionedCall2D
 conv2_c5/StatefulPartitionedCall conv2_c5/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
$__inference_bn_c3_layer_call_fn_4693

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_bn_c3_layer_call_and_return_conditional_losses_1199�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
B__inference_conv2_c5_layer_call_and_return_conditional_losses_4855

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:o
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
y
A__inference_classes_layer_call_and_return_conditional_losses_2278

inputs
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*,
_output_shapes
:����������*\
IdentityIdentityconcat:output:0*
T0*,
_output_shapes
:����������*"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������� :����������:����������:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
B__inference_conv2_c4_layer_call_and_return_conditional_losses_4761

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:  �
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: o
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:��������� X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:��������� i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�U
�
B__inference_ssd_head_layer_call_and_return_conditional_losses_2282

inputs$

model_1959:

model_1961:

model_1963:

model_1965:

model_1967:

model_1969:$

model_1971:

model_1973:

model_1975:

model_1977:

model_1979:

model_1981:$

model_1983: 

model_1985: 

model_1987: 

model_1989: 

model_1991: 

model_1993: $

model_1995:  

model_1997: 

model_1999: 

model_2001: 

model_2003: 

model_2005: $

model_2007: 

model_2009:#
	off3_2027:
	off3_2029:#
	off2_2045: 
	off2_2047:#
	off1_2063: 
	off1_2065:#
	cls3_2081:
	cls3_2083:#
	cls2_2099: 
	cls2_2101:#
	cls1_2117: 
	cls1_2119:
identity

identity_1��cls1/StatefulPartitionedCall�cls2/StatefulPartitionedCall�cls3/StatefulPartitionedCall�model/StatefulPartitionedCall�off1/StatefulPartitionedCall�off2/StatefulPartitionedCall�off3/StatefulPartitionedCall�
model/StatefulPartitionedCallStatefulPartitionedCallinputs
model_1959
model_1961
model_1963
model_1965
model_1967
model_1969
model_1971
model_1973
model_1975
model_1977
model_1979
model_1981
model_1983
model_1985
model_1987
model_1989
model_1991
model_1993
model_1995
model_1997
model_1999
model_2001
model_2003
model_2005
model_2007
model_2009*&
Tin
2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:���������   :��������� :���������*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_1444�
off3/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:2	off3_2027	off3_2029*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_off3_layer_call_and_return_conditional_losses_2026�
off2/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:1	off2_2045	off2_2047*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_off2_layer_call_and_return_conditional_losses_2044�
off1/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0	off1_2063	off1_2065*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_off1_layer_call_and_return_conditional_losses_2062�
cls3/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:2	cls3_2081	cls3_2083*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_cls3_layer_call_and_return_conditional_losses_2080�
cls2/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:1	cls2_2099	cls2_2101*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_cls2_layer_call_and_return_conditional_losses_2098�
cls1/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0	cls1_2117	cls1_2119*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_cls1_layer_call_and_return_conditional_losses_2116�
off_res3/PartitionedCallPartitionedCall%off3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_off_res3_layer_call_and_return_conditional_losses_2135�
off_res2/PartitionedCallPartitionedCall%off2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_off_res2_layer_call_and_return_conditional_losses_2150�
off_res1/PartitionedCallPartitionedCall%off1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_off_res1_layer_call_and_return_conditional_losses_2165�
cls_res3/PartitionedCallPartitionedCall%cls3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_cls_res3_layer_call_and_return_conditional_losses_2180�
cls_res2/PartitionedCallPartitionedCall%cls2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_cls_res2_layer_call_and_return_conditional_losses_2195�
cls_res1/PartitionedCallPartitionedCall%cls1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_cls_res1_layer_call_and_return_conditional_losses_2210�
off_cat1/PartitionedCallPartitionedCall!off_res1/PartitionedCall:output:0!off_res1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_off_cat1_layer_call_and_return_conditional_losses_2219�
off_cat2/PartitionedCallPartitionedCall!off_res2/PartitionedCall:output:0!off_res2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_off_cat2_layer_call_and_return_conditional_losses_2228�
off_cat3/PartitionedCallPartitionedCall!off_res3/PartitionedCall:output:0!off_res3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_off_cat3_layer_call_and_return_conditional_losses_2237�
cls_out1/PartitionedCallPartitionedCall!cls_res1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_cls_out1_layer_call_and_return_conditional_losses_2244�
cls_out2/PartitionedCallPartitionedCall!cls_res2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_cls_out2_layer_call_and_return_conditional_losses_2251�
cls_out3/PartitionedCallPartitionedCall!cls_res3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_cls_out3_layer_call_and_return_conditional_losses_2258�
offsets/PartitionedCallPartitionedCall!off_cat1/PartitionedCall:output:0!off_cat2/PartitionedCall:output:0!off_cat3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_offsets_layer_call_and_return_conditional_losses_2268�
classes/PartitionedCallPartitionedCall!cls_out1/PartitionedCall:output:0!cls_out2/PartitionedCall:output:0!cls_out3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_classes_layer_call_and_return_conditional_losses_2278t
IdentityIdentity classes/PartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������*v

Identity_1Identity offsets/PartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������*�
NoOpNoOp^cls1/StatefulPartitionedCall^cls2/StatefulPartitionedCall^cls3/StatefulPartitionedCall^model/StatefulPartitionedCall^off1/StatefulPartitionedCall^off2/StatefulPartitionedCall^off3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
cls1/StatefulPartitionedCallcls1/StatefulPartitionedCall2<
cls2/StatefulPartitionedCallcls2/StatefulPartitionedCall2<
cls3/StatefulPartitionedCallcls3/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2<
off1/StatefulPartitionedCalloff1/StatefulPartitionedCall2<
off2/StatefulPartitionedCalloff2/StatefulPartitionedCall2<
off3/StatefulPartitionedCalloff3/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
B__inference_conv2_c2_layer_call_and_return_conditional_losses_1348

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:o
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
$__inference_bn_c2_layer_call_fn_4586

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_bn_c2_layer_call_and_return_conditional_losses_1092�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�u
�
?__inference_model_layer_call_and_return_conditional_losses_4012

inputsA
'conv2_c1_conv2d_readvariableop_resource:6
(conv2_c1_biasadd_readvariableop_resource:+
bn_c1_readvariableop_resource:-
bn_c1_readvariableop_1_resource:<
.bn_c1_fusedbatchnormv3_readvariableop_resource:>
0bn_c1_fusedbatchnormv3_readvariableop_1_resource:A
'conv2_c2_conv2d_readvariableop_resource:6
(conv2_c2_biasadd_readvariableop_resource:+
bn_c2_readvariableop_resource:-
bn_c2_readvariableop_1_resource:<
.bn_c2_fusedbatchnormv3_readvariableop_resource:>
0bn_c2_fusedbatchnormv3_readvariableop_1_resource:A
'conv2_c3_conv2d_readvariableop_resource: 6
(conv2_c3_biasadd_readvariableop_resource: +
bn_c3_readvariableop_resource: -
bn_c3_readvariableop_1_resource: <
.bn_c3_fusedbatchnormv3_readvariableop_resource: >
0bn_c3_fusedbatchnormv3_readvariableop_1_resource: A
'conv2_c4_conv2d_readvariableop_resource:  6
(conv2_c4_biasadd_readvariableop_resource: +
bn_c4_readvariableop_resource: -
bn_c4_readvariableop_1_resource: <
.bn_c4_fusedbatchnormv3_readvariableop_resource: >
0bn_c4_fusedbatchnormv3_readvariableop_1_resource: A
'conv2_c5_conv2d_readvariableop_resource: 6
(conv2_c5_biasadd_readvariableop_resource:
identity

identity_1

identity_2��%bn_c1/FusedBatchNormV3/ReadVariableOp�'bn_c1/FusedBatchNormV3/ReadVariableOp_1�bn_c1/ReadVariableOp�bn_c1/ReadVariableOp_1�%bn_c2/FusedBatchNormV3/ReadVariableOp�'bn_c2/FusedBatchNormV3/ReadVariableOp_1�bn_c2/ReadVariableOp�bn_c2/ReadVariableOp_1�%bn_c3/FusedBatchNormV3/ReadVariableOp�'bn_c3/FusedBatchNormV3/ReadVariableOp_1�bn_c3/ReadVariableOp�bn_c3/ReadVariableOp_1�%bn_c4/FusedBatchNormV3/ReadVariableOp�'bn_c4/FusedBatchNormV3/ReadVariableOp_1�bn_c4/ReadVariableOp�bn_c4/ReadVariableOp_1�conv2_c1/BiasAdd/ReadVariableOp�conv2_c1/Conv2D/ReadVariableOp�conv2_c2/BiasAdd/ReadVariableOp�conv2_c2/Conv2D/ReadVariableOp�conv2_c3/BiasAdd/ReadVariableOp�conv2_c3/Conv2D/ReadVariableOp�conv2_c4/BiasAdd/ReadVariableOp�conv2_c4/Conv2D/ReadVariableOp�conv2_c5/BiasAdd/ReadVariableOp�conv2_c5/Conv2D/ReadVariableOph
conv2_c1/CastCastinputs*

DstT0*

SrcT0*1
_output_shapes
:������������
conv2_c1/Conv2D/ReadVariableOpReadVariableOp'conv2_c1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2_c1/Conv2D/CastCast&conv2_c1/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
conv2_c1/Conv2DConv2Dconv2_c1/Cast:y:0conv2_c1/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
conv2_c1/BiasAdd/ReadVariableOpReadVariableOp(conv2_c1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0z
conv2_c1/BiasAdd/CastCast'conv2_c1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
conv2_c1/BiasAddBiasAddconv2_c1/Conv2D:output:0conv2_c1/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������l
conv2_c1/ReluReluconv2_c1/BiasAdd:output:0*
T0*1
_output_shapes
:�����������n
bn_c1/ReadVariableOpReadVariableOpbn_c1_readvariableop_resource*
_output_shapes
:*
dtype0r
bn_c1/ReadVariableOp_1ReadVariableOpbn_c1_readvariableop_1_resource*
_output_shapes
:*
dtype0�
%bn_c1/FusedBatchNormV3/ReadVariableOpReadVariableOp.bn_c1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
'bn_c1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp0bn_c1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
bn_c1/FusedBatchNormV3FusedBatchNormV3conv2_c1/Relu:activations:0bn_c1/ReadVariableOp:value:0bn_c1/ReadVariableOp_1:value:0-bn_c1/FusedBatchNormV3/ReadVariableOp:value:0/bn_c1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:�����������:::::*
epsilon%o�:*
is_training( �
max_pooling2d/MaxPoolMaxPoolbn_c1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@@*
ksize
*
paddingSAME*
strides
�
conv2_c2/Conv2D/ReadVariableOpReadVariableOp'conv2_c2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2_c2/Conv2D/CastCast&conv2_c2/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
conv2_c2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0conv2_c2/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
�
conv2_c2/BiasAdd/ReadVariableOpReadVariableOp(conv2_c2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0z
conv2_c2/BiasAdd/CastCast'conv2_c2/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
conv2_c2/BiasAddBiasAddconv2_c2/Conv2D:output:0conv2_c2/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������@@j
conv2_c2/ReluReluconv2_c2/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@n
bn_c2/ReadVariableOpReadVariableOpbn_c2_readvariableop_resource*
_output_shapes
:*
dtype0r
bn_c2/ReadVariableOp_1ReadVariableOpbn_c2_readvariableop_1_resource*
_output_shapes
:*
dtype0�
%bn_c2/FusedBatchNormV3/ReadVariableOpReadVariableOp.bn_c2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
'bn_c2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp0bn_c2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
bn_c2/FusedBatchNormV3FusedBatchNormV3conv2_c2/Relu:activations:0bn_c2/ReadVariableOp:value:0bn_c2/ReadVariableOp_1:value:0-bn_c2/FusedBatchNormV3/ReadVariableOp:value:0/bn_c2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@@:::::*
epsilon%o�:*
is_training( �
max_pooling2d_1/MaxPoolMaxPoolbn_c2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������  *
ksize
*
paddingSAME*
strides
�
conv2_c3/Conv2D/ReadVariableOpReadVariableOp'conv2_c3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2_c3/Conv2D/CastCast&conv2_c3/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
conv2_c3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0conv2_c3/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
�
conv2_c3/BiasAdd/ReadVariableOpReadVariableOp(conv2_c3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0z
conv2_c3/BiasAdd/CastCast'conv2_c3/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: �
conv2_c3/BiasAddBiasAddconv2_c3/Conv2D:output:0conv2_c3/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������   j
conv2_c3/ReluReluconv2_c3/BiasAdd:output:0*
T0*/
_output_shapes
:���������   n
bn_c3/ReadVariableOpReadVariableOpbn_c3_readvariableop_resource*
_output_shapes
: *
dtype0r
bn_c3/ReadVariableOp_1ReadVariableOpbn_c3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
%bn_c3/FusedBatchNormV3/ReadVariableOpReadVariableOp.bn_c3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
'bn_c3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp0bn_c3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
bn_c3/FusedBatchNormV3FusedBatchNormV3conv2_c3/Relu:activations:0bn_c3/ReadVariableOp:value:0bn_c3/ReadVariableOp_1:value:0-bn_c3/FusedBatchNormV3/ReadVariableOp:value:0/bn_c3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
is_training( �
max_pooling2d_2/MaxPoolMaxPoolbn_c3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:��������� *
ksize
*
paddingSAME*
strides
�
conv2_c4/Conv2D/ReadVariableOpReadVariableOp'conv2_c4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
conv2_c4/Conv2D/CastCast&conv2_c4/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:  �
conv2_c4/Conv2DConv2D max_pooling2d_2/MaxPool:output:0conv2_c4/Conv2D/Cast:y:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
conv2_c4/BiasAdd/ReadVariableOpReadVariableOp(conv2_c4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0z
conv2_c4/BiasAdd/CastCast'conv2_c4/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: �
conv2_c4/BiasAddBiasAddconv2_c4/Conv2D:output:0conv2_c4/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:��������� j
conv2_c4/ReluReluconv2_c4/BiasAdd:output:0*
T0*/
_output_shapes
:��������� n
bn_c4/ReadVariableOpReadVariableOpbn_c4_readvariableop_resource*
_output_shapes
: *
dtype0r
bn_c4/ReadVariableOp_1ReadVariableOpbn_c4_readvariableop_1_resource*
_output_shapes
: *
dtype0�
%bn_c4/FusedBatchNormV3/ReadVariableOpReadVariableOp.bn_c4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
'bn_c4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp0bn_c4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
bn_c4/FusedBatchNormV3FusedBatchNormV3conv2_c4/Relu:activations:0bn_c4/ReadVariableOp:value:0bn_c4/ReadVariableOp_1:value:0-bn_c4/FusedBatchNormV3/ReadVariableOp:value:0/bn_c4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
is_training( �
max_pooling2d_3/MaxPoolMaxPoolbn_c4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:��������� *
ksize
*
paddingSAME*
strides
�
conv2_c5/Conv2D/ReadVariableOpReadVariableOp'conv2_c5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2_c5/Conv2D/CastCast&conv2_c5/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
conv2_c5/Conv2DConv2D max_pooling2d_3/MaxPool:output:0conv2_c5/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
conv2_c5/BiasAdd/ReadVariableOpReadVariableOp(conv2_c5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0z
conv2_c5/BiasAdd/CastCast'conv2_c5/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
conv2_c5/BiasAddBiasAddconv2_c5/Conv2D:output:0conv2_c5/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������j
conv2_c5/ReluReluconv2_c5/BiasAdd:output:0*
T0*/
_output_shapes
:���������q
IdentityIdentitybn_c3/FusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:���������   s

Identity_1Identitybn_c4/FusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:��������� t

Identity_2Identityconv2_c5/Relu:activations:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp&^bn_c1/FusedBatchNormV3/ReadVariableOp(^bn_c1/FusedBatchNormV3/ReadVariableOp_1^bn_c1/ReadVariableOp^bn_c1/ReadVariableOp_1&^bn_c2/FusedBatchNormV3/ReadVariableOp(^bn_c2/FusedBatchNormV3/ReadVariableOp_1^bn_c2/ReadVariableOp^bn_c2/ReadVariableOp_1&^bn_c3/FusedBatchNormV3/ReadVariableOp(^bn_c3/FusedBatchNormV3/ReadVariableOp_1^bn_c3/ReadVariableOp^bn_c3/ReadVariableOp_1&^bn_c4/FusedBatchNormV3/ReadVariableOp(^bn_c4/FusedBatchNormV3/ReadVariableOp_1^bn_c4/ReadVariableOp^bn_c4/ReadVariableOp_1 ^conv2_c1/BiasAdd/ReadVariableOp^conv2_c1/Conv2D/ReadVariableOp ^conv2_c2/BiasAdd/ReadVariableOp^conv2_c2/Conv2D/ReadVariableOp ^conv2_c3/BiasAdd/ReadVariableOp^conv2_c3/Conv2D/ReadVariableOp ^conv2_c4/BiasAdd/ReadVariableOp^conv2_c4/Conv2D/ReadVariableOp ^conv2_c5/BiasAdd/ReadVariableOp^conv2_c5/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%bn_c1/FusedBatchNormV3/ReadVariableOp%bn_c1/FusedBatchNormV3/ReadVariableOp2R
'bn_c1/FusedBatchNormV3/ReadVariableOp_1'bn_c1/FusedBatchNormV3/ReadVariableOp_12,
bn_c1/ReadVariableOpbn_c1/ReadVariableOp20
bn_c1/ReadVariableOp_1bn_c1/ReadVariableOp_12N
%bn_c2/FusedBatchNormV3/ReadVariableOp%bn_c2/FusedBatchNormV3/ReadVariableOp2R
'bn_c2/FusedBatchNormV3/ReadVariableOp_1'bn_c2/FusedBatchNormV3/ReadVariableOp_12,
bn_c2/ReadVariableOpbn_c2/ReadVariableOp20
bn_c2/ReadVariableOp_1bn_c2/ReadVariableOp_12N
%bn_c3/FusedBatchNormV3/ReadVariableOp%bn_c3/FusedBatchNormV3/ReadVariableOp2R
'bn_c3/FusedBatchNormV3/ReadVariableOp_1'bn_c3/FusedBatchNormV3/ReadVariableOp_12,
bn_c3/ReadVariableOpbn_c3/ReadVariableOp20
bn_c3/ReadVariableOp_1bn_c3/ReadVariableOp_12N
%bn_c4/FusedBatchNormV3/ReadVariableOp%bn_c4/FusedBatchNormV3/ReadVariableOp2R
'bn_c4/FusedBatchNormV3/ReadVariableOp_1'bn_c4/FusedBatchNormV3/ReadVariableOp_12,
bn_c4/ReadVariableOpbn_c4/ReadVariableOp20
bn_c4/ReadVariableOp_1bn_c4/ReadVariableOp_12B
conv2_c1/BiasAdd/ReadVariableOpconv2_c1/BiasAdd/ReadVariableOp2@
conv2_c1/Conv2D/ReadVariableOpconv2_c1/Conv2D/ReadVariableOp2B
conv2_c2/BiasAdd/ReadVariableOpconv2_c2/BiasAdd/ReadVariableOp2@
conv2_c2/Conv2D/ReadVariableOpconv2_c2/Conv2D/ReadVariableOp2B
conv2_c3/BiasAdd/ReadVariableOpconv2_c3/BiasAdd/ReadVariableOp2@
conv2_c3/Conv2D/ReadVariableOpconv2_c3/Conv2D/ReadVariableOp2B
conv2_c4/BiasAdd/ReadVariableOpconv2_c4/BiasAdd/ReadVariableOp2@
conv2_c4/Conv2D/ReadVariableOpconv2_c4/Conv2D/ReadVariableOp2B
conv2_c5/BiasAdd/ReadVariableOpconv2_c5/BiasAdd/ReadVariableOp2@
conv2_c5/Conv2D/ReadVariableOpconv2_c5/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
C
'__inference_cls_res3_layer_call_fn_4291

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_cls_res3_layer_call_and_return_conditional_losses_2180e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

^
B__inference_off_res1_layer_call_and_return_conditional_losses_4322

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:i
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:���������� ]
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:���������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������  :W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
{
A__inference_classes_layer_call_and_return_conditional_losses_4442
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*,
_output_shapes
:����������*\
IdentityIdentityconcat:output:0*
T0*,
_output_shapes
:����������*"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������� :����������:����������:V R
,
_output_shapes
:���������� 
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs_1:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs_2
�
�
#__inference_off1_layer_call_fn_4196

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_off1_layer_call_and_return_conditional_losses_2062w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
�
'__inference_conv2_c1_layer_call_fn_4466

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2_c1_layer_call_and_return_conditional_losses_1319y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
?__inference_bn_c2_layer_call_and_return_conditional_losses_4617

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�

^
B__inference_off_res3_layer_call_and_return_conditional_losses_2135

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:i
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:����������]
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
?__inference_bn_c1_layer_call_and_return_conditional_losses_4541

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
?__inference_bn_c1_layer_call_and_return_conditional_losses_1016

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
?__inference_bn_c4_layer_call_and_return_conditional_losses_4805

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
B__inference_conv2_c1_layer_call_and_return_conditional_losses_1319

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:q
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:�����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
?__inference_bn_c1_layer_call_and_return_conditional_losses_4523

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
n
B__inference_off_cat2_layer_call_and_return_conditional_losses_4414
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :|
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*,
_output_shapes
:����������\
IdentityIdentityconcat:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������:����������:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs_1
�
e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1143

inputs
identity�
MaxPoolMaxPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
>__inference_off1_layer_call_and_return_conditional_losses_4208

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:o
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������  g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������  w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�U
�
B__inference_ssd_head_layer_call_and_return_conditional_losses_2703

inputs$

model_2602:

model_2604:

model_2606:

model_2608:

model_2610:

model_2612:$

model_2614:

model_2616:

model_2618:

model_2620:

model_2622:

model_2624:$

model_2626: 

model_2628: 

model_2630: 

model_2632: 

model_2634: 

model_2636: $

model_2638:  

model_2640: 

model_2642: 

model_2644: 

model_2646: 

model_2648: $

model_2650: 

model_2652:#
	off3_2657:
	off3_2659:#
	off2_2662: 
	off2_2664:#
	off1_2667: 
	off1_2669:#
	cls3_2672:
	cls3_2674:#
	cls2_2677: 
	cls2_2679:#
	cls1_2682: 
	cls1_2684:
identity

identity_1��cls1/StatefulPartitionedCall�cls2/StatefulPartitionedCall�cls3/StatefulPartitionedCall�model/StatefulPartitionedCall�off1/StatefulPartitionedCall�off2/StatefulPartitionedCall�off3/StatefulPartitionedCall�
model/StatefulPartitionedCallStatefulPartitionedCallinputs
model_2602
model_2604
model_2606
model_2608
model_2610
model_2612
model_2614
model_2616
model_2618
model_2620
model_2622
model_2624
model_2626
model_2628
model_2630
model_2632
model_2634
model_2636
model_2638
model_2640
model_2642
model_2644
model_2646
model_2648
model_2650
model_2652*&
Tin
2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:���������   :��������� :���������*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_1688�
off3/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:2	off3_2657	off3_2659*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_off3_layer_call_and_return_conditional_losses_2026�
off2/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:1	off2_2662	off2_2664*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_off2_layer_call_and_return_conditional_losses_2044�
off1/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0	off1_2667	off1_2669*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_off1_layer_call_and_return_conditional_losses_2062�
cls3/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:2	cls3_2672	cls3_2674*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_cls3_layer_call_and_return_conditional_losses_2080�
cls2/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:1	cls2_2677	cls2_2679*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_cls2_layer_call_and_return_conditional_losses_2098�
cls1/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0	cls1_2682	cls1_2684*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_cls1_layer_call_and_return_conditional_losses_2116�
off_res3/PartitionedCallPartitionedCall%off3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_off_res3_layer_call_and_return_conditional_losses_2135�
off_res2/PartitionedCallPartitionedCall%off2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_off_res2_layer_call_and_return_conditional_losses_2150�
off_res1/PartitionedCallPartitionedCall%off1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_off_res1_layer_call_and_return_conditional_losses_2165�
cls_res3/PartitionedCallPartitionedCall%cls3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_cls_res3_layer_call_and_return_conditional_losses_2180�
cls_res2/PartitionedCallPartitionedCall%cls2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_cls_res2_layer_call_and_return_conditional_losses_2195�
cls_res1/PartitionedCallPartitionedCall%cls1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_cls_res1_layer_call_and_return_conditional_losses_2210�
off_cat1/PartitionedCallPartitionedCall!off_res1/PartitionedCall:output:0!off_res1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_off_cat1_layer_call_and_return_conditional_losses_2219�
off_cat2/PartitionedCallPartitionedCall!off_res2/PartitionedCall:output:0!off_res2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_off_cat2_layer_call_and_return_conditional_losses_2228�
off_cat3/PartitionedCallPartitionedCall!off_res3/PartitionedCall:output:0!off_res3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_off_cat3_layer_call_and_return_conditional_losses_2237�
cls_out1/PartitionedCallPartitionedCall!cls_res1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_cls_out1_layer_call_and_return_conditional_losses_2244�
cls_out2/PartitionedCallPartitionedCall!cls_res2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_cls_out2_layer_call_and_return_conditional_losses_2251�
cls_out3/PartitionedCallPartitionedCall!cls_res3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_cls_out3_layer_call_and_return_conditional_losses_2258�
offsets/PartitionedCallPartitionedCall!off_cat1/PartitionedCall:output:0!off_cat2/PartitionedCall:output:0!off_cat3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_offsets_layer_call_and_return_conditional_losses_2268�
classes/PartitionedCallPartitionedCall!cls_out1/PartitionedCall:output:0!cls_out2/PartitionedCall:output:0!cls_out3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_classes_layer_call_and_return_conditional_losses_2278t
IdentityIdentity classes/PartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������*v

Identity_1Identity offsets/PartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������*�
NoOpNoOp^cls1/StatefulPartitionedCall^cls2/StatefulPartitionedCall^cls3/StatefulPartitionedCall^model/StatefulPartitionedCall^off1/StatefulPartitionedCall^off2/StatefulPartitionedCall^off3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
cls1/StatefulPartitionedCallcls1/StatefulPartitionedCall2<
cls2/StatefulPartitionedCallcls2/StatefulPartitionedCall2<
cls3/StatefulPartitionedCallcls3/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2<
off1/StatefulPartitionedCalloff1/StatefulPartitionedCall2<
off2/StatefulPartitionedCalloff2/StatefulPartitionedCall2<
off3/StatefulPartitionedCalloff3/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
?__inference_bn_c2_layer_call_and_return_conditional_losses_1123

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
>__inference_cls2_layer_call_and_return_conditional_losses_4166

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:o
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
?__inference_bn_c2_layer_call_and_return_conditional_losses_4635

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
$__inference_bn_c1_layer_call_fn_4492

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_bn_c1_layer_call_and_return_conditional_losses_1016�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
#__inference_off3_layer_call_fn_4238

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_off3_layer_call_and_return_conditional_losses_2026w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�	
"__inference_signature_wrapper_3160
input_1!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17:  

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23: 

unknown_24:$

unknown_25:

unknown_26:$

unknown_27: 

unknown_28:$

unknown_29: 

unknown_30:$

unknown_31:

unknown_32:$

unknown_33: 

unknown_34:$

unknown_35: 

unknown_36:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *D
_output_shapes2
0:����������*:����������**H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8� *'
f"R 
__inference__wrapped_model_994t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������*v

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*,
_output_shapes
:����������*`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
�
B__inference_conv2_c2_layer_call_and_return_conditional_losses_4573

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:o
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
C
'__inference_cls_res2_layer_call_fn_4273

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_cls_res2_layer_call_and_return_conditional_losses_2195e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
^
B__inference_cls_out3_layer_call_and_return_conditional_losses_4388

inputs
identityQ
SoftmaxSoftmaxinputs*
T0*,
_output_shapes
:����������^
IdentityIdentitySoftmax:softmax:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
$__inference_model_layer_call_fn_3900

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17:  

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23: 

unknown_24:
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:���������   :��������� :���������*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_1688w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������   y

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:��������� y

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
S
'__inference_off_cat3_layer_call_fn_4420
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_off_cat3_layer_call_and_return_conditional_losses_2237e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������:����������:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs_1
�
e
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_4833

inputs
identity�
MaxPoolMaxPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
`
&__inference_offsets_layer_call_fn_4449
inputs_0
inputs_1
inputs_2
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_offsets_layer_call_and_return_conditional_losses_2268e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������*"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������� :����������:����������:V R
,
_output_shapes
:���������� 
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs_1:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs_2
�

^
B__inference_off_res3_layer_call_and_return_conditional_losses_4358

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:i
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:����������]
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
B__inference_conv2_c3_layer_call_and_return_conditional_losses_1377

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: o
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������   X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������   i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
C
'__inference_off_res2_layer_call_fn_4327

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_off_res2_layer_call_and_return_conditional_losses_2150e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
?__inference_model_layer_call_and_return_conditional_losses_4124

inputsA
'conv2_c1_conv2d_readvariableop_resource:6
(conv2_c1_biasadd_readvariableop_resource:+
bn_c1_readvariableop_resource:-
bn_c1_readvariableop_1_resource:<
.bn_c1_fusedbatchnormv3_readvariableop_resource:>
0bn_c1_fusedbatchnormv3_readvariableop_1_resource:A
'conv2_c2_conv2d_readvariableop_resource:6
(conv2_c2_biasadd_readvariableop_resource:+
bn_c2_readvariableop_resource:-
bn_c2_readvariableop_1_resource:<
.bn_c2_fusedbatchnormv3_readvariableop_resource:>
0bn_c2_fusedbatchnormv3_readvariableop_1_resource:A
'conv2_c3_conv2d_readvariableop_resource: 6
(conv2_c3_biasadd_readvariableop_resource: +
bn_c3_readvariableop_resource: -
bn_c3_readvariableop_1_resource: <
.bn_c3_fusedbatchnormv3_readvariableop_resource: >
0bn_c3_fusedbatchnormv3_readvariableop_1_resource: A
'conv2_c4_conv2d_readvariableop_resource:  6
(conv2_c4_biasadd_readvariableop_resource: +
bn_c4_readvariableop_resource: -
bn_c4_readvariableop_1_resource: <
.bn_c4_fusedbatchnormv3_readvariableop_resource: >
0bn_c4_fusedbatchnormv3_readvariableop_1_resource: A
'conv2_c5_conv2d_readvariableop_resource: 6
(conv2_c5_biasadd_readvariableop_resource:
identity

identity_1

identity_2��bn_c1/AssignNewValue�bn_c1/AssignNewValue_1�%bn_c1/FusedBatchNormV3/ReadVariableOp�'bn_c1/FusedBatchNormV3/ReadVariableOp_1�bn_c1/ReadVariableOp�bn_c1/ReadVariableOp_1�bn_c2/AssignNewValue�bn_c2/AssignNewValue_1�%bn_c2/FusedBatchNormV3/ReadVariableOp�'bn_c2/FusedBatchNormV3/ReadVariableOp_1�bn_c2/ReadVariableOp�bn_c2/ReadVariableOp_1�bn_c3/AssignNewValue�bn_c3/AssignNewValue_1�%bn_c3/FusedBatchNormV3/ReadVariableOp�'bn_c3/FusedBatchNormV3/ReadVariableOp_1�bn_c3/ReadVariableOp�bn_c3/ReadVariableOp_1�bn_c4/AssignNewValue�bn_c4/AssignNewValue_1�%bn_c4/FusedBatchNormV3/ReadVariableOp�'bn_c4/FusedBatchNormV3/ReadVariableOp_1�bn_c4/ReadVariableOp�bn_c4/ReadVariableOp_1�conv2_c1/BiasAdd/ReadVariableOp�conv2_c1/Conv2D/ReadVariableOp�conv2_c2/BiasAdd/ReadVariableOp�conv2_c2/Conv2D/ReadVariableOp�conv2_c3/BiasAdd/ReadVariableOp�conv2_c3/Conv2D/ReadVariableOp�conv2_c4/BiasAdd/ReadVariableOp�conv2_c4/Conv2D/ReadVariableOp�conv2_c5/BiasAdd/ReadVariableOp�conv2_c5/Conv2D/ReadVariableOph
conv2_c1/CastCastinputs*

DstT0*

SrcT0*1
_output_shapes
:������������
conv2_c1/Conv2D/ReadVariableOpReadVariableOp'conv2_c1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2_c1/Conv2D/CastCast&conv2_c1/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
conv2_c1/Conv2DConv2Dconv2_c1/Cast:y:0conv2_c1/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
conv2_c1/BiasAdd/ReadVariableOpReadVariableOp(conv2_c1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0z
conv2_c1/BiasAdd/CastCast'conv2_c1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
conv2_c1/BiasAddBiasAddconv2_c1/Conv2D:output:0conv2_c1/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������l
conv2_c1/ReluReluconv2_c1/BiasAdd:output:0*
T0*1
_output_shapes
:�����������n
bn_c1/ReadVariableOpReadVariableOpbn_c1_readvariableop_resource*
_output_shapes
:*
dtype0r
bn_c1/ReadVariableOp_1ReadVariableOpbn_c1_readvariableop_1_resource*
_output_shapes
:*
dtype0�
%bn_c1/FusedBatchNormV3/ReadVariableOpReadVariableOp.bn_c1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
'bn_c1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp0bn_c1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
bn_c1/FusedBatchNormV3FusedBatchNormV3conv2_c1/Relu:activations:0bn_c1/ReadVariableOp:value:0bn_c1/ReadVariableOp_1:value:0-bn_c1/FusedBatchNormV3/ReadVariableOp:value:0/bn_c1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:�����������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
bn_c1/AssignNewValueAssignVariableOp.bn_c1_fusedbatchnormv3_readvariableop_resource#bn_c1/FusedBatchNormV3:batch_mean:0&^bn_c1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
bn_c1/AssignNewValue_1AssignVariableOp0bn_c1_fusedbatchnormv3_readvariableop_1_resource'bn_c1/FusedBatchNormV3:batch_variance:0(^bn_c1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
max_pooling2d/MaxPoolMaxPoolbn_c1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@@*
ksize
*
paddingSAME*
strides
�
conv2_c2/Conv2D/ReadVariableOpReadVariableOp'conv2_c2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2_c2/Conv2D/CastCast&conv2_c2/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
conv2_c2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0conv2_c2/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
�
conv2_c2/BiasAdd/ReadVariableOpReadVariableOp(conv2_c2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0z
conv2_c2/BiasAdd/CastCast'conv2_c2/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
conv2_c2/BiasAddBiasAddconv2_c2/Conv2D:output:0conv2_c2/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������@@j
conv2_c2/ReluReluconv2_c2/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@n
bn_c2/ReadVariableOpReadVariableOpbn_c2_readvariableop_resource*
_output_shapes
:*
dtype0r
bn_c2/ReadVariableOp_1ReadVariableOpbn_c2_readvariableop_1_resource*
_output_shapes
:*
dtype0�
%bn_c2/FusedBatchNormV3/ReadVariableOpReadVariableOp.bn_c2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
'bn_c2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp0bn_c2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
bn_c2/FusedBatchNormV3FusedBatchNormV3conv2_c2/Relu:activations:0bn_c2/ReadVariableOp:value:0bn_c2/ReadVariableOp_1:value:0-bn_c2/FusedBatchNormV3/ReadVariableOp:value:0/bn_c2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@@:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
bn_c2/AssignNewValueAssignVariableOp.bn_c2_fusedbatchnormv3_readvariableop_resource#bn_c2/FusedBatchNormV3:batch_mean:0&^bn_c2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
bn_c2/AssignNewValue_1AssignVariableOp0bn_c2_fusedbatchnormv3_readvariableop_1_resource'bn_c2/FusedBatchNormV3:batch_variance:0(^bn_c2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
max_pooling2d_1/MaxPoolMaxPoolbn_c2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������  *
ksize
*
paddingSAME*
strides
�
conv2_c3/Conv2D/ReadVariableOpReadVariableOp'conv2_c3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2_c3/Conv2D/CastCast&conv2_c3/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
conv2_c3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0conv2_c3/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
�
conv2_c3/BiasAdd/ReadVariableOpReadVariableOp(conv2_c3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0z
conv2_c3/BiasAdd/CastCast'conv2_c3/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: �
conv2_c3/BiasAddBiasAddconv2_c3/Conv2D:output:0conv2_c3/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������   j
conv2_c3/ReluReluconv2_c3/BiasAdd:output:0*
T0*/
_output_shapes
:���������   n
bn_c3/ReadVariableOpReadVariableOpbn_c3_readvariableop_resource*
_output_shapes
: *
dtype0r
bn_c3/ReadVariableOp_1ReadVariableOpbn_c3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
%bn_c3/FusedBatchNormV3/ReadVariableOpReadVariableOp.bn_c3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
'bn_c3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp0bn_c3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
bn_c3/FusedBatchNormV3FusedBatchNormV3conv2_c3/Relu:activations:0bn_c3/ReadVariableOp:value:0bn_c3/ReadVariableOp_1:value:0-bn_c3/FusedBatchNormV3/ReadVariableOp:value:0/bn_c3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
bn_c3/AssignNewValueAssignVariableOp.bn_c3_fusedbatchnormv3_readvariableop_resource#bn_c3/FusedBatchNormV3:batch_mean:0&^bn_c3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
bn_c3/AssignNewValue_1AssignVariableOp0bn_c3_fusedbatchnormv3_readvariableop_1_resource'bn_c3/FusedBatchNormV3:batch_variance:0(^bn_c3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
max_pooling2d_2/MaxPoolMaxPoolbn_c3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:��������� *
ksize
*
paddingSAME*
strides
�
conv2_c4/Conv2D/ReadVariableOpReadVariableOp'conv2_c4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
conv2_c4/Conv2D/CastCast&conv2_c4/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:  �
conv2_c4/Conv2DConv2D max_pooling2d_2/MaxPool:output:0conv2_c4/Conv2D/Cast:y:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
conv2_c4/BiasAdd/ReadVariableOpReadVariableOp(conv2_c4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0z
conv2_c4/BiasAdd/CastCast'conv2_c4/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: �
conv2_c4/BiasAddBiasAddconv2_c4/Conv2D:output:0conv2_c4/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:��������� j
conv2_c4/ReluReluconv2_c4/BiasAdd:output:0*
T0*/
_output_shapes
:��������� n
bn_c4/ReadVariableOpReadVariableOpbn_c4_readvariableop_resource*
_output_shapes
: *
dtype0r
bn_c4/ReadVariableOp_1ReadVariableOpbn_c4_readvariableop_1_resource*
_output_shapes
: *
dtype0�
%bn_c4/FusedBatchNormV3/ReadVariableOpReadVariableOp.bn_c4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
'bn_c4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp0bn_c4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
bn_c4/FusedBatchNormV3FusedBatchNormV3conv2_c4/Relu:activations:0bn_c4/ReadVariableOp:value:0bn_c4/ReadVariableOp_1:value:0-bn_c4/FusedBatchNormV3/ReadVariableOp:value:0/bn_c4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
bn_c4/AssignNewValueAssignVariableOp.bn_c4_fusedbatchnormv3_readvariableop_resource#bn_c4/FusedBatchNormV3:batch_mean:0&^bn_c4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
bn_c4/AssignNewValue_1AssignVariableOp0bn_c4_fusedbatchnormv3_readvariableop_1_resource'bn_c4/FusedBatchNormV3:batch_variance:0(^bn_c4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
max_pooling2d_3/MaxPoolMaxPoolbn_c4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:��������� *
ksize
*
paddingSAME*
strides
�
conv2_c5/Conv2D/ReadVariableOpReadVariableOp'conv2_c5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2_c5/Conv2D/CastCast&conv2_c5/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
conv2_c5/Conv2DConv2D max_pooling2d_3/MaxPool:output:0conv2_c5/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
conv2_c5/BiasAdd/ReadVariableOpReadVariableOp(conv2_c5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0z
conv2_c5/BiasAdd/CastCast'conv2_c5/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
conv2_c5/BiasAddBiasAddconv2_c5/Conv2D:output:0conv2_c5/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������j
conv2_c5/ReluReluconv2_c5/BiasAdd:output:0*
T0*/
_output_shapes
:���������q
IdentityIdentitybn_c3/FusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:���������   s

Identity_1Identitybn_c4/FusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:��������� t

Identity_2Identityconv2_c5/Relu:activations:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^bn_c1/AssignNewValue^bn_c1/AssignNewValue_1&^bn_c1/FusedBatchNormV3/ReadVariableOp(^bn_c1/FusedBatchNormV3/ReadVariableOp_1^bn_c1/ReadVariableOp^bn_c1/ReadVariableOp_1^bn_c2/AssignNewValue^bn_c2/AssignNewValue_1&^bn_c2/FusedBatchNormV3/ReadVariableOp(^bn_c2/FusedBatchNormV3/ReadVariableOp_1^bn_c2/ReadVariableOp^bn_c2/ReadVariableOp_1^bn_c3/AssignNewValue^bn_c3/AssignNewValue_1&^bn_c3/FusedBatchNormV3/ReadVariableOp(^bn_c3/FusedBatchNormV3/ReadVariableOp_1^bn_c3/ReadVariableOp^bn_c3/ReadVariableOp_1^bn_c4/AssignNewValue^bn_c4/AssignNewValue_1&^bn_c4/FusedBatchNormV3/ReadVariableOp(^bn_c4/FusedBatchNormV3/ReadVariableOp_1^bn_c4/ReadVariableOp^bn_c4/ReadVariableOp_1 ^conv2_c1/BiasAdd/ReadVariableOp^conv2_c1/Conv2D/ReadVariableOp ^conv2_c2/BiasAdd/ReadVariableOp^conv2_c2/Conv2D/ReadVariableOp ^conv2_c3/BiasAdd/ReadVariableOp^conv2_c3/Conv2D/ReadVariableOp ^conv2_c4/BiasAdd/ReadVariableOp^conv2_c4/Conv2D/ReadVariableOp ^conv2_c5/BiasAdd/ReadVariableOp^conv2_c5/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : 2,
bn_c1/AssignNewValuebn_c1/AssignNewValue20
bn_c1/AssignNewValue_1bn_c1/AssignNewValue_12N
%bn_c1/FusedBatchNormV3/ReadVariableOp%bn_c1/FusedBatchNormV3/ReadVariableOp2R
'bn_c1/FusedBatchNormV3/ReadVariableOp_1'bn_c1/FusedBatchNormV3/ReadVariableOp_12,
bn_c1/ReadVariableOpbn_c1/ReadVariableOp20
bn_c1/ReadVariableOp_1bn_c1/ReadVariableOp_12,
bn_c2/AssignNewValuebn_c2/AssignNewValue20
bn_c2/AssignNewValue_1bn_c2/AssignNewValue_12N
%bn_c2/FusedBatchNormV3/ReadVariableOp%bn_c2/FusedBatchNormV3/ReadVariableOp2R
'bn_c2/FusedBatchNormV3/ReadVariableOp_1'bn_c2/FusedBatchNormV3/ReadVariableOp_12,
bn_c2/ReadVariableOpbn_c2/ReadVariableOp20
bn_c2/ReadVariableOp_1bn_c2/ReadVariableOp_12,
bn_c3/AssignNewValuebn_c3/AssignNewValue20
bn_c3/AssignNewValue_1bn_c3/AssignNewValue_12N
%bn_c3/FusedBatchNormV3/ReadVariableOp%bn_c3/FusedBatchNormV3/ReadVariableOp2R
'bn_c3/FusedBatchNormV3/ReadVariableOp_1'bn_c3/FusedBatchNormV3/ReadVariableOp_12,
bn_c3/ReadVariableOpbn_c3/ReadVariableOp20
bn_c3/ReadVariableOp_1bn_c3/ReadVariableOp_12,
bn_c4/AssignNewValuebn_c4/AssignNewValue20
bn_c4/AssignNewValue_1bn_c4/AssignNewValue_12N
%bn_c4/FusedBatchNormV3/ReadVariableOp%bn_c4/FusedBatchNormV3/ReadVariableOp2R
'bn_c4/FusedBatchNormV3/ReadVariableOp_1'bn_c4/FusedBatchNormV3/ReadVariableOp_12,
bn_c4/ReadVariableOpbn_c4/ReadVariableOp20
bn_c4/ReadVariableOp_1bn_c4/ReadVariableOp_12B
conv2_c1/BiasAdd/ReadVariableOpconv2_c1/BiasAdd/ReadVariableOp2@
conv2_c1/Conv2D/ReadVariableOpconv2_c1/Conv2D/ReadVariableOp2B
conv2_c2/BiasAdd/ReadVariableOpconv2_c2/BiasAdd/ReadVariableOp2@
conv2_c2/Conv2D/ReadVariableOpconv2_c2/Conv2D/ReadVariableOp2B
conv2_c3/BiasAdd/ReadVariableOpconv2_c3/BiasAdd/ReadVariableOp2@
conv2_c3/Conv2D/ReadVariableOpconv2_c3/Conv2D/ReadVariableOp2B
conv2_c4/BiasAdd/ReadVariableOpconv2_c4/BiasAdd/ReadVariableOp2@
conv2_c4/Conv2D/ReadVariableOpconv2_c4/Conv2D/ReadVariableOp2B
conv2_c5/BiasAdd/ReadVariableOpconv2_c5/BiasAdd/ReadVariableOp2@
conv2_c5/Conv2D/ReadVariableOpconv2_c5/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
��
�
B__inference_ssd_head_layer_call_and_return_conditional_losses_3552

inputsG
-model_conv2_c1_conv2d_readvariableop_resource:<
.model_conv2_c1_biasadd_readvariableop_resource:1
#model_bn_c1_readvariableop_resource:3
%model_bn_c1_readvariableop_1_resource:B
4model_bn_c1_fusedbatchnormv3_readvariableop_resource:D
6model_bn_c1_fusedbatchnormv3_readvariableop_1_resource:G
-model_conv2_c2_conv2d_readvariableop_resource:<
.model_conv2_c2_biasadd_readvariableop_resource:1
#model_bn_c2_readvariableop_resource:3
%model_bn_c2_readvariableop_1_resource:B
4model_bn_c2_fusedbatchnormv3_readvariableop_resource:D
6model_bn_c2_fusedbatchnormv3_readvariableop_1_resource:G
-model_conv2_c3_conv2d_readvariableop_resource: <
.model_conv2_c3_biasadd_readvariableop_resource: 1
#model_bn_c3_readvariableop_resource: 3
%model_bn_c3_readvariableop_1_resource: B
4model_bn_c3_fusedbatchnormv3_readvariableop_resource: D
6model_bn_c3_fusedbatchnormv3_readvariableop_1_resource: G
-model_conv2_c4_conv2d_readvariableop_resource:  <
.model_conv2_c4_biasadd_readvariableop_resource: 1
#model_bn_c4_readvariableop_resource: 3
%model_bn_c4_readvariableop_1_resource: B
4model_bn_c4_fusedbatchnormv3_readvariableop_resource: D
6model_bn_c4_fusedbatchnormv3_readvariableop_1_resource: G
-model_conv2_c5_conv2d_readvariableop_resource: <
.model_conv2_c5_biasadd_readvariableop_resource:=
#off3_conv2d_readvariableop_resource:2
$off3_biasadd_readvariableop_resource:=
#off2_conv2d_readvariableop_resource: 2
$off2_biasadd_readvariableop_resource:=
#off1_conv2d_readvariableop_resource: 2
$off1_biasadd_readvariableop_resource:=
#cls3_conv2d_readvariableop_resource:2
$cls3_biasadd_readvariableop_resource:=
#cls2_conv2d_readvariableop_resource: 2
$cls2_biasadd_readvariableop_resource:=
#cls1_conv2d_readvariableop_resource: 2
$cls1_biasadd_readvariableop_resource:
identity

identity_1��cls1/BiasAdd/ReadVariableOp�cls1/Conv2D/ReadVariableOp�cls2/BiasAdd/ReadVariableOp�cls2/Conv2D/ReadVariableOp�cls3/BiasAdd/ReadVariableOp�cls3/Conv2D/ReadVariableOp�+model/bn_c1/FusedBatchNormV3/ReadVariableOp�-model/bn_c1/FusedBatchNormV3/ReadVariableOp_1�model/bn_c1/ReadVariableOp�model/bn_c1/ReadVariableOp_1�+model/bn_c2/FusedBatchNormV3/ReadVariableOp�-model/bn_c2/FusedBatchNormV3/ReadVariableOp_1�model/bn_c2/ReadVariableOp�model/bn_c2/ReadVariableOp_1�+model/bn_c3/FusedBatchNormV3/ReadVariableOp�-model/bn_c3/FusedBatchNormV3/ReadVariableOp_1�model/bn_c3/ReadVariableOp�model/bn_c3/ReadVariableOp_1�+model/bn_c4/FusedBatchNormV3/ReadVariableOp�-model/bn_c4/FusedBatchNormV3/ReadVariableOp_1�model/bn_c4/ReadVariableOp�model/bn_c4/ReadVariableOp_1�%model/conv2_c1/BiasAdd/ReadVariableOp�$model/conv2_c1/Conv2D/ReadVariableOp�%model/conv2_c2/BiasAdd/ReadVariableOp�$model/conv2_c2/Conv2D/ReadVariableOp�%model/conv2_c3/BiasAdd/ReadVariableOp�$model/conv2_c3/Conv2D/ReadVariableOp�%model/conv2_c4/BiasAdd/ReadVariableOp�$model/conv2_c4/Conv2D/ReadVariableOp�%model/conv2_c5/BiasAdd/ReadVariableOp�$model/conv2_c5/Conv2D/ReadVariableOp�off1/BiasAdd/ReadVariableOp�off1/Conv2D/ReadVariableOp�off2/BiasAdd/ReadVariableOp�off2/Conv2D/ReadVariableOp�off3/BiasAdd/ReadVariableOp�off3/Conv2D/ReadVariableOpn
model/conv2_c1/CastCastinputs*

DstT0*

SrcT0*1
_output_shapes
:������������
$model/conv2_c1/Conv2D/ReadVariableOpReadVariableOp-model_conv2_c1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
model/conv2_c1/Conv2D/CastCast,model/conv2_c1/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
model/conv2_c1/Conv2DConv2Dmodel/conv2_c1/Cast:y:0model/conv2_c1/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
%model/conv2_c1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2_c1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv2_c1/BiasAdd/CastCast-model/conv2_c1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
model/conv2_c1/BiasAddBiasAddmodel/conv2_c1/Conv2D:output:0model/conv2_c1/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������x
model/conv2_c1/ReluRelumodel/conv2_c1/BiasAdd:output:0*
T0*1
_output_shapes
:�����������z
model/bn_c1/ReadVariableOpReadVariableOp#model_bn_c1_readvariableop_resource*
_output_shapes
:*
dtype0~
model/bn_c1/ReadVariableOp_1ReadVariableOp%model_bn_c1_readvariableop_1_resource*
_output_shapes
:*
dtype0�
+model/bn_c1/FusedBatchNormV3/ReadVariableOpReadVariableOp4model_bn_c1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
-model/bn_c1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6model_bn_c1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
model/bn_c1/FusedBatchNormV3FusedBatchNormV3!model/conv2_c1/Relu:activations:0"model/bn_c1/ReadVariableOp:value:0$model/bn_c1/ReadVariableOp_1:value:03model/bn_c1/FusedBatchNormV3/ReadVariableOp:value:05model/bn_c1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:�����������:::::*
epsilon%o�:*
is_training( �
model/max_pooling2d/MaxPoolMaxPool model/bn_c1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@@*
ksize
*
paddingSAME*
strides
�
$model/conv2_c2/Conv2D/ReadVariableOpReadVariableOp-model_conv2_c2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
model/conv2_c2/Conv2D/CastCast,model/conv2_c2/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
model/conv2_c2/Conv2DConv2D$model/max_pooling2d/MaxPool:output:0model/conv2_c2/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
�
%model/conv2_c2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2_c2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv2_c2/BiasAdd/CastCast-model/conv2_c2/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
model/conv2_c2/BiasAddBiasAddmodel/conv2_c2/Conv2D:output:0model/conv2_c2/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������@@v
model/conv2_c2/ReluRelumodel/conv2_c2/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@z
model/bn_c2/ReadVariableOpReadVariableOp#model_bn_c2_readvariableop_resource*
_output_shapes
:*
dtype0~
model/bn_c2/ReadVariableOp_1ReadVariableOp%model_bn_c2_readvariableop_1_resource*
_output_shapes
:*
dtype0�
+model/bn_c2/FusedBatchNormV3/ReadVariableOpReadVariableOp4model_bn_c2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
-model/bn_c2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6model_bn_c2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
model/bn_c2/FusedBatchNormV3FusedBatchNormV3!model/conv2_c2/Relu:activations:0"model/bn_c2/ReadVariableOp:value:0$model/bn_c2/ReadVariableOp_1:value:03model/bn_c2/FusedBatchNormV3/ReadVariableOp:value:05model/bn_c2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@@:::::*
epsilon%o�:*
is_training( �
model/max_pooling2d_1/MaxPoolMaxPool model/bn_c2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������  *
ksize
*
paddingSAME*
strides
�
$model/conv2_c3/Conv2D/ReadVariableOpReadVariableOp-model_conv2_c3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
model/conv2_c3/Conv2D/CastCast,model/conv2_c3/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
model/conv2_c3/Conv2DConv2D&model/max_pooling2d_1/MaxPool:output:0model/conv2_c3/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
�
%model/conv2_c3/BiasAdd/ReadVariableOpReadVariableOp.model_conv2_c3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model/conv2_c3/BiasAdd/CastCast-model/conv2_c3/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: �
model/conv2_c3/BiasAddBiasAddmodel/conv2_c3/Conv2D:output:0model/conv2_c3/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������   v
model/conv2_c3/ReluRelumodel/conv2_c3/BiasAdd:output:0*
T0*/
_output_shapes
:���������   z
model/bn_c3/ReadVariableOpReadVariableOp#model_bn_c3_readvariableop_resource*
_output_shapes
: *
dtype0~
model/bn_c3/ReadVariableOp_1ReadVariableOp%model_bn_c3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
+model/bn_c3/FusedBatchNormV3/ReadVariableOpReadVariableOp4model_bn_c3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
-model/bn_c3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6model_bn_c3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
model/bn_c3/FusedBatchNormV3FusedBatchNormV3!model/conv2_c3/Relu:activations:0"model/bn_c3/ReadVariableOp:value:0$model/bn_c3/ReadVariableOp_1:value:03model/bn_c3/FusedBatchNormV3/ReadVariableOp:value:05model/bn_c3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
is_training( �
model/max_pooling2d_2/MaxPoolMaxPool model/bn_c3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:��������� *
ksize
*
paddingSAME*
strides
�
$model/conv2_c4/Conv2D/ReadVariableOpReadVariableOp-model_conv2_c4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
model/conv2_c4/Conv2D/CastCast,model/conv2_c4/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:  �
model/conv2_c4/Conv2DConv2D&model/max_pooling2d_2/MaxPool:output:0model/conv2_c4/Conv2D/Cast:y:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
%model/conv2_c4/BiasAdd/ReadVariableOpReadVariableOp.model_conv2_c4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model/conv2_c4/BiasAdd/CastCast-model/conv2_c4/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: �
model/conv2_c4/BiasAddBiasAddmodel/conv2_c4/Conv2D:output:0model/conv2_c4/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:��������� v
model/conv2_c4/ReluRelumodel/conv2_c4/BiasAdd:output:0*
T0*/
_output_shapes
:��������� z
model/bn_c4/ReadVariableOpReadVariableOp#model_bn_c4_readvariableop_resource*
_output_shapes
: *
dtype0~
model/bn_c4/ReadVariableOp_1ReadVariableOp%model_bn_c4_readvariableop_1_resource*
_output_shapes
: *
dtype0�
+model/bn_c4/FusedBatchNormV3/ReadVariableOpReadVariableOp4model_bn_c4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
-model/bn_c4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6model_bn_c4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
model/bn_c4/FusedBatchNormV3FusedBatchNormV3!model/conv2_c4/Relu:activations:0"model/bn_c4/ReadVariableOp:value:0$model/bn_c4/ReadVariableOp_1:value:03model/bn_c4/FusedBatchNormV3/ReadVariableOp:value:05model/bn_c4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
is_training( �
model/max_pooling2d_3/MaxPoolMaxPool model/bn_c4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:��������� *
ksize
*
paddingSAME*
strides
�
$model/conv2_c5/Conv2D/ReadVariableOpReadVariableOp-model_conv2_c5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
model/conv2_c5/Conv2D/CastCast,model/conv2_c5/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
model/conv2_c5/Conv2DConv2D&model/max_pooling2d_3/MaxPool:output:0model/conv2_c5/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
%model/conv2_c5/BiasAdd/ReadVariableOpReadVariableOp.model_conv2_c5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv2_c5/BiasAdd/CastCast-model/conv2_c5/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
model/conv2_c5/BiasAddBiasAddmodel/conv2_c5/Conv2D:output:0model/conv2_c5/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������v
model/conv2_c5/ReluRelumodel/conv2_c5/BiasAdd:output:0*
T0*/
_output_shapes
:����������
off3/Conv2D/ReadVariableOpReadVariableOp#off3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0|
off3/Conv2D/CastCast"off3/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
off3/Conv2DConv2D!model/conv2_c5/Relu:activations:0off3/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
|
off3/BiasAdd/ReadVariableOpReadVariableOp$off3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0r
off3/BiasAdd/CastCast#off3/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:~
off3/BiasAddBiasAddoff3/Conv2D:output:0off3/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:����������
off2/Conv2D/ReadVariableOpReadVariableOp#off2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0|
off2/Conv2D/CastCast"off2/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
off2/Conv2DConv2D model/bn_c4/FusedBatchNormV3:y:0off2/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
|
off2/BiasAdd/ReadVariableOpReadVariableOp$off2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0r
off2/BiasAdd/CastCast#off2/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:~
off2/BiasAddBiasAddoff2/Conv2D:output:0off2/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:����������
off1/Conv2D/ReadVariableOpReadVariableOp#off1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0|
off1/Conv2D/CastCast"off1/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
off1/Conv2DConv2D model/bn_c3/FusedBatchNormV3:y:0off1/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
|
off1/BiasAdd/ReadVariableOpReadVariableOp$off1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0r
off1/BiasAdd/CastCast#off1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:~
off1/BiasAddBiasAddoff1/Conv2D:output:0off1/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������  �
cls3/Conv2D/ReadVariableOpReadVariableOp#cls3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0|
cls3/Conv2D/CastCast"cls3/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
cls3/Conv2DConv2D!model/conv2_c5/Relu:activations:0cls3/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
|
cls3/BiasAdd/ReadVariableOpReadVariableOp$cls3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0r
cls3/BiasAdd/CastCast#cls3/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:~
cls3/BiasAddBiasAddcls3/Conv2D:output:0cls3/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:����������
cls2/Conv2D/ReadVariableOpReadVariableOp#cls2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0|
cls2/Conv2D/CastCast"cls2/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
cls2/Conv2DConv2D model/bn_c4/FusedBatchNormV3:y:0cls2/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
|
cls2/BiasAdd/ReadVariableOpReadVariableOp$cls2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0r
cls2/BiasAdd/CastCast#cls2/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:~
cls2/BiasAddBiasAddcls2/Conv2D:output:0cls2/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:����������
cls1/Conv2D/ReadVariableOpReadVariableOp#cls1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0|
cls1/Conv2D/CastCast"cls1/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
cls1/Conv2DConv2D model/bn_c3/FusedBatchNormV3:y:0cls1/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
|
cls1/BiasAdd/ReadVariableOpReadVariableOp$cls1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0r
cls1/BiasAdd/CastCast#cls1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:~
cls1/BiasAddBiasAddcls1/Conv2D:output:0cls1/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������  S
off_res3/ShapeShapeoff3/BiasAdd:output:0*
T0*
_output_shapes
:f
off_res3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
off_res3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
off_res3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
off_res3/strided_sliceStridedSliceoff_res3/Shape:output:0%off_res3/strided_slice/stack:output:0'off_res3/strided_slice/stack_1:output:0'off_res3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
off_res3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������Z
off_res3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
off_res3/Reshape/shapePackoff_res3/strided_slice:output:0!off_res3/Reshape/shape/1:output:0!off_res3/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
off_res3/ReshapeReshapeoff3/BiasAdd:output:0off_res3/Reshape/shape:output:0*
T0*,
_output_shapes
:����������S
off_res2/ShapeShapeoff2/BiasAdd:output:0*
T0*
_output_shapes
:f
off_res2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
off_res2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
off_res2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
off_res2/strided_sliceStridedSliceoff_res2/Shape:output:0%off_res2/strided_slice/stack:output:0'off_res2/strided_slice/stack_1:output:0'off_res2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
off_res2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������Z
off_res2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
off_res2/Reshape/shapePackoff_res2/strided_slice:output:0!off_res2/Reshape/shape/1:output:0!off_res2/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
off_res2/ReshapeReshapeoff2/BiasAdd:output:0off_res2/Reshape/shape:output:0*
T0*,
_output_shapes
:����������S
off_res1/ShapeShapeoff1/BiasAdd:output:0*
T0*
_output_shapes
:f
off_res1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
off_res1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
off_res1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
off_res1/strided_sliceStridedSliceoff_res1/Shape:output:0%off_res1/strided_slice/stack:output:0'off_res1/strided_slice/stack_1:output:0'off_res1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
off_res1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������Z
off_res1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
off_res1/Reshape/shapePackoff_res1/strided_slice:output:0!off_res1/Reshape/shape/1:output:0!off_res1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
off_res1/ReshapeReshapeoff1/BiasAdd:output:0off_res1/Reshape/shape:output:0*
T0*,
_output_shapes
:���������� S
cls_res3/ShapeShapecls3/BiasAdd:output:0*
T0*
_output_shapes
:f
cls_res3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
cls_res3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
cls_res3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
cls_res3/strided_sliceStridedSlicecls_res3/Shape:output:0%cls_res3/strided_slice/stack:output:0'cls_res3/strided_slice/stack_1:output:0'cls_res3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
cls_res3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������Z
cls_res3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
cls_res3/Reshape/shapePackcls_res3/strided_slice:output:0!cls_res3/Reshape/shape/1:output:0!cls_res3/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
cls_res3/ReshapeReshapecls3/BiasAdd:output:0cls_res3/Reshape/shape:output:0*
T0*,
_output_shapes
:����������S
cls_res2/ShapeShapecls2/BiasAdd:output:0*
T0*
_output_shapes
:f
cls_res2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
cls_res2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
cls_res2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
cls_res2/strided_sliceStridedSlicecls_res2/Shape:output:0%cls_res2/strided_slice/stack:output:0'cls_res2/strided_slice/stack_1:output:0'cls_res2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
cls_res2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������Z
cls_res2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
cls_res2/Reshape/shapePackcls_res2/strided_slice:output:0!cls_res2/Reshape/shape/1:output:0!cls_res2/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
cls_res2/ReshapeReshapecls2/BiasAdd:output:0cls_res2/Reshape/shape:output:0*
T0*,
_output_shapes
:����������S
cls_res1/ShapeShapecls1/BiasAdd:output:0*
T0*
_output_shapes
:f
cls_res1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
cls_res1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
cls_res1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
cls_res1/strided_sliceStridedSlicecls_res1/Shape:output:0%cls_res1/strided_slice/stack:output:0'cls_res1/strided_slice/stack_1:output:0'cls_res1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
cls_res1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������Z
cls_res1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
cls_res1/Reshape/shapePackcls_res1/strided_slice:output:0!cls_res1/Reshape/shape/1:output:0!cls_res1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
cls_res1/ReshapeReshapecls1/BiasAdd:output:0cls_res1/Reshape/shape:output:0*
T0*,
_output_shapes
:���������� V
off_cat1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
off_cat1/concatConcatV2off_res1/Reshape:output:0off_res1/Reshape:output:0off_cat1/concat/axis:output:0*
N*
T0*,
_output_shapes
:���������� V
off_cat2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
off_cat2/concatConcatV2off_res2/Reshape:output:0off_res2/Reshape:output:0off_cat2/concat/axis:output:0*
N*
T0*,
_output_shapes
:����������V
off_cat3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
off_cat3/concatConcatV2off_res3/Reshape:output:0off_res3/Reshape:output:0off_cat3/concat/axis:output:0*
N*
T0*,
_output_shapes
:����������m
cls_out1/SoftmaxSoftmaxcls_res1/Reshape:output:0*
T0*,
_output_shapes
:���������� m
cls_out2/SoftmaxSoftmaxcls_res2/Reshape:output:0*
T0*,
_output_shapes
:����������m
cls_out3/SoftmaxSoftmaxcls_res3/Reshape:output:0*
T0*,
_output_shapes
:����������U
offsets/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
offsets/concatConcatV2off_cat1/concat:output:0off_cat2/concat:output:0off_cat3/concat:output:0offsets/concat/axis:output:0*
N*
T0*,
_output_shapes
:����������*U
classes/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
classes/concatConcatV2cls_out1/Softmax:softmax:0cls_out2/Softmax:softmax:0cls_out3/Softmax:softmax:0classes/concat/axis:output:0*
N*
T0*,
_output_shapes
:����������*k
IdentityIdentityclasses/concat:output:0^NoOp*
T0*,
_output_shapes
:����������*m

Identity_1Identityoffsets/concat:output:0^NoOp*
T0*,
_output_shapes
:����������*�
NoOpNoOp^cls1/BiasAdd/ReadVariableOp^cls1/Conv2D/ReadVariableOp^cls2/BiasAdd/ReadVariableOp^cls2/Conv2D/ReadVariableOp^cls3/BiasAdd/ReadVariableOp^cls3/Conv2D/ReadVariableOp,^model/bn_c1/FusedBatchNormV3/ReadVariableOp.^model/bn_c1/FusedBatchNormV3/ReadVariableOp_1^model/bn_c1/ReadVariableOp^model/bn_c1/ReadVariableOp_1,^model/bn_c2/FusedBatchNormV3/ReadVariableOp.^model/bn_c2/FusedBatchNormV3/ReadVariableOp_1^model/bn_c2/ReadVariableOp^model/bn_c2/ReadVariableOp_1,^model/bn_c3/FusedBatchNormV3/ReadVariableOp.^model/bn_c3/FusedBatchNormV3/ReadVariableOp_1^model/bn_c3/ReadVariableOp^model/bn_c3/ReadVariableOp_1,^model/bn_c4/FusedBatchNormV3/ReadVariableOp.^model/bn_c4/FusedBatchNormV3/ReadVariableOp_1^model/bn_c4/ReadVariableOp^model/bn_c4/ReadVariableOp_1&^model/conv2_c1/BiasAdd/ReadVariableOp%^model/conv2_c1/Conv2D/ReadVariableOp&^model/conv2_c2/BiasAdd/ReadVariableOp%^model/conv2_c2/Conv2D/ReadVariableOp&^model/conv2_c3/BiasAdd/ReadVariableOp%^model/conv2_c3/Conv2D/ReadVariableOp&^model/conv2_c4/BiasAdd/ReadVariableOp%^model/conv2_c4/Conv2D/ReadVariableOp&^model/conv2_c5/BiasAdd/ReadVariableOp%^model/conv2_c5/Conv2D/ReadVariableOp^off1/BiasAdd/ReadVariableOp^off1/Conv2D/ReadVariableOp^off2/BiasAdd/ReadVariableOp^off2/Conv2D/ReadVariableOp^off3/BiasAdd/ReadVariableOp^off3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
cls1/BiasAdd/ReadVariableOpcls1/BiasAdd/ReadVariableOp28
cls1/Conv2D/ReadVariableOpcls1/Conv2D/ReadVariableOp2:
cls2/BiasAdd/ReadVariableOpcls2/BiasAdd/ReadVariableOp28
cls2/Conv2D/ReadVariableOpcls2/Conv2D/ReadVariableOp2:
cls3/BiasAdd/ReadVariableOpcls3/BiasAdd/ReadVariableOp28
cls3/Conv2D/ReadVariableOpcls3/Conv2D/ReadVariableOp2Z
+model/bn_c1/FusedBatchNormV3/ReadVariableOp+model/bn_c1/FusedBatchNormV3/ReadVariableOp2^
-model/bn_c1/FusedBatchNormV3/ReadVariableOp_1-model/bn_c1/FusedBatchNormV3/ReadVariableOp_128
model/bn_c1/ReadVariableOpmodel/bn_c1/ReadVariableOp2<
model/bn_c1/ReadVariableOp_1model/bn_c1/ReadVariableOp_12Z
+model/bn_c2/FusedBatchNormV3/ReadVariableOp+model/bn_c2/FusedBatchNormV3/ReadVariableOp2^
-model/bn_c2/FusedBatchNormV3/ReadVariableOp_1-model/bn_c2/FusedBatchNormV3/ReadVariableOp_128
model/bn_c2/ReadVariableOpmodel/bn_c2/ReadVariableOp2<
model/bn_c2/ReadVariableOp_1model/bn_c2/ReadVariableOp_12Z
+model/bn_c3/FusedBatchNormV3/ReadVariableOp+model/bn_c3/FusedBatchNormV3/ReadVariableOp2^
-model/bn_c3/FusedBatchNormV3/ReadVariableOp_1-model/bn_c3/FusedBatchNormV3/ReadVariableOp_128
model/bn_c3/ReadVariableOpmodel/bn_c3/ReadVariableOp2<
model/bn_c3/ReadVariableOp_1model/bn_c3/ReadVariableOp_12Z
+model/bn_c4/FusedBatchNormV3/ReadVariableOp+model/bn_c4/FusedBatchNormV3/ReadVariableOp2^
-model/bn_c4/FusedBatchNormV3/ReadVariableOp_1-model/bn_c4/FusedBatchNormV3/ReadVariableOp_128
model/bn_c4/ReadVariableOpmodel/bn_c4/ReadVariableOp2<
model/bn_c4/ReadVariableOp_1model/bn_c4/ReadVariableOp_12N
%model/conv2_c1/BiasAdd/ReadVariableOp%model/conv2_c1/BiasAdd/ReadVariableOp2L
$model/conv2_c1/Conv2D/ReadVariableOp$model/conv2_c1/Conv2D/ReadVariableOp2N
%model/conv2_c2/BiasAdd/ReadVariableOp%model/conv2_c2/BiasAdd/ReadVariableOp2L
$model/conv2_c2/Conv2D/ReadVariableOp$model/conv2_c2/Conv2D/ReadVariableOp2N
%model/conv2_c3/BiasAdd/ReadVariableOp%model/conv2_c3/BiasAdd/ReadVariableOp2L
$model/conv2_c3/Conv2D/ReadVariableOp$model/conv2_c3/Conv2D/ReadVariableOp2N
%model/conv2_c4/BiasAdd/ReadVariableOp%model/conv2_c4/BiasAdd/ReadVariableOp2L
$model/conv2_c4/Conv2D/ReadVariableOp$model/conv2_c4/Conv2D/ReadVariableOp2N
%model/conv2_c5/BiasAdd/ReadVariableOp%model/conv2_c5/BiasAdd/ReadVariableOp2L
$model/conv2_c5/Conv2D/ReadVariableOp$model/conv2_c5/Conv2D/ReadVariableOp2:
off1/BiasAdd/ReadVariableOpoff1/BiasAdd/ReadVariableOp28
off1/Conv2D/ReadVariableOpoff1/Conv2D/ReadVariableOp2:
off2/BiasAdd/ReadVariableOpoff2/BiasAdd/ReadVariableOp28
off2/Conv2D/ReadVariableOpoff2/Conv2D/ReadVariableOp2:
off3/BiasAdd/ReadVariableOpoff3/BiasAdd/ReadVariableOp28
off3/Conv2D/ReadVariableOpoff3/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
?__inference_bn_c4_layer_call_and_return_conditional_losses_1244

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�

^
B__inference_cls_res2_layer_call_and_return_conditional_losses_4286

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:i
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:����������]
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
S
'__inference_off_cat2_layer_call_fn_4407
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_off_cat2_layer_call_and_return_conditional_losses_2228e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������:����������:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs_1
�
�
?__inference_bn_c1_layer_call_and_return_conditional_losses_1047

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
>__inference_cls2_layer_call_and_return_conditional_losses_2098

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: �
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:o
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
>__inference_off3_layer_call_and_return_conditional_losses_2026

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:o
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
#__inference_cls2_layer_call_fn_4154

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_cls2_layer_call_and_return_conditional_losses_2098w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
C
'__inference_cls_out2_layer_call_fn_4373

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_cls_out2_layer_call_and_return_conditional_losses_2251e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
E
input_1:
serving_default_input_1:0�����������@
classes5
StatefulPartitionedCall:0����������*@
offsets5
StatefulPartitionedCall:1����������*tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
layer-0
 layer_with_weights-0
 layer-1
!layer_with_weights-1
!layer-2
"layer-3
#layer_with_weights-2
#layer-4
$layer_with_weights-3
$layer-5
%layer-6
&layer_with_weights-4
&layer-7
'layer_with_weights-5
'layer-8
(layer-9
)layer_with_weights-6
)layer-10
*layer_with_weights-7
*layer-11
+layer-12
,layer_with_weights-8
,layer-13
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses"
_tf_keras_network
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

9kernel
:bias
 ;_jit_compiled_convolution_op"
_tf_keras_layer
�
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses

Bkernel
Cbias
 D_jit_compiled_convolution_op"
_tf_keras_layer
�
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses

Kkernel
Lbias
 M_jit_compiled_convolution_op"
_tf_keras_layer
�
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses

Tkernel
Ubias
 V_jit_compiled_convolution_op"
_tf_keras_layer
�
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses

]kernel
^bias
 __jit_compiled_convolution_op"
_tf_keras_layer
�
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses

fkernel
gbias
 h_jit_compiled_convolution_op"
_tf_keras_layer
�
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses"
_tf_keras_layer
�
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses"
_tf_keras_layer
�
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses"
_tf_keras_layer
�
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
926
:27
B28
C29
K30
L31
T32
U33
]34
^35
f36
g37"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
918
:19
B20
C21
K22
L23
T24
U25
]26
^27
f28
g29"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
'__inference_ssd_head_layer_call_fn_2363
'__inference_ssd_head_layer_call_fn_3243
'__inference_ssd_head_layer_call_fn_3326
'__inference_ssd_head_layer_call_fn_2867�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
B__inference_ssd_head_layer_call_and_return_conditional_losses_3552
B__inference_ssd_head_layer_call_and_return_conditional_losses_3778
B__inference_ssd_head_layer_call_and_return_conditional_losses_2971
B__inference_ssd_head_layer_call_and_return_conditional_losses_3075�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
__inference__wrapped_model_994input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
-
�serving_default"
signature_map
"
_tf_keras_input_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
$__inference_model_layer_call_fn_1503
$__inference_model_layer_call_fn_3839
$__inference_model_layer_call_fn_3900
$__inference_model_layer_call_fn_1808�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
?__inference_model_layer_call_and_return_conditional_losses_4012
?__inference_model_layer_call_and_return_conditional_losses_4124
?__inference_model_layer_call_and_return_conditional_losses_1880
?__inference_model_layer_call_and_return_conditional_losses_1952�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
#__inference_cls1_layer_call_fn_4133�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
>__inference_cls1_layer_call_and_return_conditional_losses_4145�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
%:# 2cls1/kernel
:2	cls1/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
#__inference_cls2_layer_call_fn_4154�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
>__inference_cls2_layer_call_and_return_conditional_losses_4166�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
%:# 2cls2/kernel
:2	cls2/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
#__inference_cls3_layer_call_fn_4175�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
>__inference_cls3_layer_call_and_return_conditional_losses_4187�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
%:#2cls3/kernel
:2	cls3/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
T0
U1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
#__inference_off1_layer_call_fn_4196�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
>__inference_off1_layer_call_and_return_conditional_losses_4208�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
%:# 2off1/kernel
:2	off1/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
#__inference_off2_layer_call_fn_4217�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
>__inference_off2_layer_call_and_return_conditional_losses_4229�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
%:# 2off2/kernel
:2	off2/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
f0
g1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
#__inference_off3_layer_call_fn_4238�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
>__inference_off3_layer_call_and_return_conditional_losses_4250�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
%:#2off3/kernel
:2	off3/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_cls_res1_layer_call_fn_4255�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_cls_res1_layer_call_and_return_conditional_losses_4268�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_cls_res2_layer_call_fn_4273�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_cls_res2_layer_call_and_return_conditional_losses_4286�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_cls_res3_layer_call_fn_4291�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_cls_res3_layer_call_and_return_conditional_losses_4304�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_off_res1_layer_call_fn_4309�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_off_res1_layer_call_and_return_conditional_losses_4322�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_off_res2_layer_call_fn_4327�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_off_res2_layer_call_and_return_conditional_losses_4340�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_off_res3_layer_call_fn_4345�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_off_res3_layer_call_and_return_conditional_losses_4358�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_cls_out1_layer_call_fn_4363�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_cls_out1_layer_call_and_return_conditional_losses_4368�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_cls_out2_layer_call_fn_4373�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_cls_out2_layer_call_and_return_conditional_losses_4378�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_cls_out3_layer_call_fn_4383�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_cls_out3_layer_call_and_return_conditional_losses_4388�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_off_cat1_layer_call_fn_4394�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_off_cat1_layer_call_and_return_conditional_losses_4401�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_off_cat2_layer_call_fn_4407�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_off_cat2_layer_call_and_return_conditional_losses_4414�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_off_cat3_layer_call_fn_4420�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_off_cat3_layer_call_and_return_conditional_losses_4427�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_classes_layer_call_fn_4434�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_classes_layer_call_and_return_conditional_losses_4442�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_offsets_layer_call_fn_4449�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_offsets_layer_call_and_return_conditional_losses_4457�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
):'2conv2_c1/kernel
:2conv2_c1/bias
:2bn_c1/gamma
:2
bn_c1/beta
!: (2bn_c1/moving_mean
%:# (2bn_c1/moving_variance
):'2conv2_c2/kernel
:2conv2_c2/bias
:2bn_c2/gamma
:2
bn_c2/beta
!: (2bn_c2/moving_mean
%:# (2bn_c2/moving_variance
):' 2conv2_c3/kernel
: 2conv2_c3/bias
: 2bn_c3/gamma
: 2
bn_c3/beta
!:  (2bn_c3/moving_mean
%:#  (2bn_c3/moving_variance
):'  2conv2_c4/kernel
: 2conv2_c4/bias
: 2bn_c4/gamma
: 2
bn_c4/beta
!:  (2bn_c4/moving_mean
%:#  (2bn_c4/moving_variance
):' 2conv2_c5/kernel
:2conv2_c5/bias
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_ssd_head_layer_call_fn_2363input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_ssd_head_layer_call_fn_3243inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_ssd_head_layer_call_fn_3326inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_ssd_head_layer_call_fn_2867input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_ssd_head_layer_call_and_return_conditional_losses_3552inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_ssd_head_layer_call_and_return_conditional_losses_3778inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_ssd_head_layer_call_and_return_conditional_losses_2971input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_ssd_head_layer_call_and_return_conditional_losses_3075input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference_signature_wrapper_3160input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_conv2_c1_layer_call_fn_4466�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_conv2_c1_layer_call_and_return_conditional_losses_4479�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
$__inference_bn_c1_layer_call_fn_4492
$__inference_bn_c1_layer_call_fn_4505�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
?__inference_bn_c1_layer_call_and_return_conditional_losses_4523
?__inference_bn_c1_layer_call_and_return_conditional_losses_4541�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_max_pooling2d_layer_call_fn_4546�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4551�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_conv2_c2_layer_call_fn_4560�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_conv2_c2_layer_call_and_return_conditional_losses_4573�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
$__inference_bn_c2_layer_call_fn_4586
$__inference_bn_c2_layer_call_fn_4599�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
?__inference_bn_c2_layer_call_and_return_conditional_losses_4617
?__inference_bn_c2_layer_call_and_return_conditional_losses_4635�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_max_pooling2d_1_layer_call_fn_4640�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4645�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_conv2_c3_layer_call_fn_4654�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_conv2_c3_layer_call_and_return_conditional_losses_4667�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
$__inference_bn_c3_layer_call_fn_4680
$__inference_bn_c3_layer_call_fn_4693�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
?__inference_bn_c3_layer_call_and_return_conditional_losses_4711
?__inference_bn_c3_layer_call_and_return_conditional_losses_4729�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_max_pooling2d_2_layer_call_fn_4734�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_4739�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_conv2_c4_layer_call_fn_4748�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_conv2_c4_layer_call_and_return_conditional_losses_4761�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
$__inference_bn_c4_layer_call_fn_4774
$__inference_bn_c4_layer_call_fn_4787�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
?__inference_bn_c4_layer_call_and_return_conditional_losses_4805
?__inference_bn_c4_layer_call_and_return_conditional_losses_4823�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_max_pooling2d_3_layer_call_fn_4828�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_4833�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_conv2_c5_layer_call_fn_4842�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_conv2_c5_layer_call_and_return_conditional_losses_4855�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
�
0
 1
!2
"3
#4
$5
%6
&7
'8
(9
)10
*11
+12
,13"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_model_layer_call_fn_1503	input_map"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_model_layer_call_fn_3839inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_model_layer_call_fn_3900inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_model_layer_call_fn_1808	input_map"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_model_layer_call_and_return_conditional_losses_4012inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_model_layer_call_and_return_conditional_losses_4124inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_model_layer_call_and_return_conditional_losses_1880	input_map"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_model_layer_call_and_return_conditional_losses_1952	input_map"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
#__inference_cls1_layer_call_fn_4133inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_cls1_layer_call_and_return_conditional_losses_4145inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
#__inference_cls2_layer_call_fn_4154inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_cls2_layer_call_and_return_conditional_losses_4166inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
#__inference_cls3_layer_call_fn_4175inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_cls3_layer_call_and_return_conditional_losses_4187inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
#__inference_off1_layer_call_fn_4196inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_off1_layer_call_and_return_conditional_losses_4208inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
#__inference_off2_layer_call_fn_4217inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_off2_layer_call_and_return_conditional_losses_4229inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
#__inference_off3_layer_call_fn_4238inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_off3_layer_call_and_return_conditional_losses_4250inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_cls_res1_layer_call_fn_4255inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_cls_res1_layer_call_and_return_conditional_losses_4268inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_cls_res2_layer_call_fn_4273inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_cls_res2_layer_call_and_return_conditional_losses_4286inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_cls_res3_layer_call_fn_4291inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_cls_res3_layer_call_and_return_conditional_losses_4304inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_off_res1_layer_call_fn_4309inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_off_res1_layer_call_and_return_conditional_losses_4322inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_off_res2_layer_call_fn_4327inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_off_res2_layer_call_and_return_conditional_losses_4340inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_off_res3_layer_call_fn_4345inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_off_res3_layer_call_and_return_conditional_losses_4358inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_cls_out1_layer_call_fn_4363inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_cls_out1_layer_call_and_return_conditional_losses_4368inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_cls_out2_layer_call_fn_4373inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_cls_out2_layer_call_and_return_conditional_losses_4378inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_cls_out3_layer_call_fn_4383inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_cls_out3_layer_call_and_return_conditional_losses_4388inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_off_cat1_layer_call_fn_4394inputs_0inputs_1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_off_cat1_layer_call_and_return_conditional_losses_4401inputs_0inputs_1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_off_cat2_layer_call_fn_4407inputs_0inputs_1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_off_cat2_layer_call_and_return_conditional_losses_4414inputs_0inputs_1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_off_cat3_layer_call_fn_4420inputs_0inputs_1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_off_cat3_layer_call_and_return_conditional_losses_4427inputs_0inputs_1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_classes_layer_call_fn_4434inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_classes_layer_call_and_return_conditional_losses_4442inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_offsets_layer_call_fn_4449inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_offsets_layer_call_and_return_conditional_losses_4457inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_conv2_c1_layer_call_fn_4466inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_conv2_c1_layer_call_and_return_conditional_losses_4479inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_bn_c1_layer_call_fn_4492inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_bn_c1_layer_call_fn_4505inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_bn_c1_layer_call_and_return_conditional_losses_4523inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_bn_c1_layer_call_and_return_conditional_losses_4541inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_max_pooling2d_layer_call_fn_4546inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4551inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_conv2_c2_layer_call_fn_4560inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_conv2_c2_layer_call_and_return_conditional_losses_4573inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_bn_c2_layer_call_fn_4586inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_bn_c2_layer_call_fn_4599inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_bn_c2_layer_call_and_return_conditional_losses_4617inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_bn_c2_layer_call_and_return_conditional_losses_4635inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_max_pooling2d_1_layer_call_fn_4640inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4645inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_conv2_c3_layer_call_fn_4654inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_conv2_c3_layer_call_and_return_conditional_losses_4667inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_bn_c3_layer_call_fn_4680inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_bn_c3_layer_call_fn_4693inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_bn_c3_layer_call_and_return_conditional_losses_4711inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_bn_c3_layer_call_and_return_conditional_losses_4729inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_max_pooling2d_2_layer_call_fn_4734inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_4739inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_conv2_c4_layer_call_fn_4748inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_conv2_c4_layer_call_and_return_conditional_losses_4761inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_bn_c4_layer_call_fn_4774inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_bn_c4_layer_call_fn_4787inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_bn_c4_layer_call_and_return_conditional_losses_4805inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_bn_c4_layer_call_and_return_conditional_losses_4823inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_max_pooling2d_3_layer_call_fn_4828inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_4833inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_conv2_c5_layer_call_fn_4842inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_conv2_c5_layer_call_and_return_conditional_losses_4855inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
__inference__wrapped_model_994�@��������������������������fg]^TUKLBC9::�7
0�-
+�(
input_1�����������
� "i�f
1
classes&�#
classes����������*
1
offsets&�#
offsets����������*�
?__inference_bn_c1_layer_call_and_return_conditional_losses_4523�����M�J
C�@
:�7
inputs+���������������������������
p 
� "F�C
<�9
tensor_0+���������������������������
� �
?__inference_bn_c1_layer_call_and_return_conditional_losses_4541�����M�J
C�@
:�7
inputs+���������������������������
p
� "F�C
<�9
tensor_0+���������������������������
� �
$__inference_bn_c1_layer_call_fn_4492�����M�J
C�@
:�7
inputs+���������������������������
p 
� ";�8
unknown+����������������������������
$__inference_bn_c1_layer_call_fn_4505�����M�J
C�@
:�7
inputs+���������������������������
p
� ";�8
unknown+����������������������������
?__inference_bn_c2_layer_call_and_return_conditional_losses_4617�����M�J
C�@
:�7
inputs+���������������������������
p 
� "F�C
<�9
tensor_0+���������������������������
� �
?__inference_bn_c2_layer_call_and_return_conditional_losses_4635�����M�J
C�@
:�7
inputs+���������������������������
p
� "F�C
<�9
tensor_0+���������������������������
� �
$__inference_bn_c2_layer_call_fn_4586�����M�J
C�@
:�7
inputs+���������������������������
p 
� ";�8
unknown+����������������������������
$__inference_bn_c2_layer_call_fn_4599�����M�J
C�@
:�7
inputs+���������������������������
p
� ";�8
unknown+����������������������������
?__inference_bn_c3_layer_call_and_return_conditional_losses_4711�����M�J
C�@
:�7
inputs+��������������������������� 
p 
� "F�C
<�9
tensor_0+��������������������������� 
� �
?__inference_bn_c3_layer_call_and_return_conditional_losses_4729�����M�J
C�@
:�7
inputs+��������������������������� 
p
� "F�C
<�9
tensor_0+��������������������������� 
� �
$__inference_bn_c3_layer_call_fn_4680�����M�J
C�@
:�7
inputs+��������������������������� 
p 
� ";�8
unknown+��������������������������� �
$__inference_bn_c3_layer_call_fn_4693�����M�J
C�@
:�7
inputs+��������������������������� 
p
� ";�8
unknown+��������������������������� �
?__inference_bn_c4_layer_call_and_return_conditional_losses_4805�����M�J
C�@
:�7
inputs+��������������������������� 
p 
� "F�C
<�9
tensor_0+��������������������������� 
� �
?__inference_bn_c4_layer_call_and_return_conditional_losses_4823�����M�J
C�@
:�7
inputs+��������������������������� 
p
� "F�C
<�9
tensor_0+��������������������������� 
� �
$__inference_bn_c4_layer_call_fn_4774�����M�J
C�@
:�7
inputs+��������������������������� 
p 
� ";�8
unknown+��������������������������� �
$__inference_bn_c4_layer_call_fn_4787�����M�J
C�@
:�7
inputs+��������������������������� 
p
� ";�8
unknown+��������������������������� �
A__inference_classes_layer_call_and_return_conditional_losses_4442����
���
~�{
'�$
inputs_0���������� 
'�$
inputs_1����������
'�$
inputs_2����������
� "1�.
'�$
tensor_0����������*
� �
&__inference_classes_layer_call_fn_4434����
���
~�{
'�$
inputs_0���������� 
'�$
inputs_1����������
'�$
inputs_2����������
� "&�#
unknown����������*�
>__inference_cls1_layer_call_and_return_conditional_losses_4145s9:7�4
-�*
(�%
inputs���������   
� "4�1
*�'
tensor_0���������  
� �
#__inference_cls1_layer_call_fn_4133h9:7�4
-�*
(�%
inputs���������   
� ")�&
unknown���������  �
>__inference_cls2_layer_call_and_return_conditional_losses_4166sBC7�4
-�*
(�%
inputs��������� 
� "4�1
*�'
tensor_0���������
� �
#__inference_cls2_layer_call_fn_4154hBC7�4
-�*
(�%
inputs��������� 
� ")�&
unknown����������
>__inference_cls3_layer_call_and_return_conditional_losses_4187sKL7�4
-�*
(�%
inputs���������
� "4�1
*�'
tensor_0���������
� �
#__inference_cls3_layer_call_fn_4175hKL7�4
-�*
(�%
inputs���������
� ")�&
unknown����������
B__inference_cls_out1_layer_call_and_return_conditional_losses_4368i4�1
*�'
%�"
inputs���������� 
� "1�.
'�$
tensor_0���������� 
� �
'__inference_cls_out1_layer_call_fn_4363^4�1
*�'
%�"
inputs���������� 
� "&�#
unknown���������� �
B__inference_cls_out2_layer_call_and_return_conditional_losses_4378i4�1
*�'
%�"
inputs����������
� "1�.
'�$
tensor_0����������
� �
'__inference_cls_out2_layer_call_fn_4373^4�1
*�'
%�"
inputs����������
� "&�#
unknown�����������
B__inference_cls_out3_layer_call_and_return_conditional_losses_4388i4�1
*�'
%�"
inputs����������
� "1�.
'�$
tensor_0����������
� �
'__inference_cls_out3_layer_call_fn_4383^4�1
*�'
%�"
inputs����������
� "&�#
unknown�����������
B__inference_cls_res1_layer_call_and_return_conditional_losses_4268l7�4
-�*
(�%
inputs���������  
� "1�.
'�$
tensor_0���������� 
� �
'__inference_cls_res1_layer_call_fn_4255a7�4
-�*
(�%
inputs���������  
� "&�#
unknown���������� �
B__inference_cls_res2_layer_call_and_return_conditional_losses_4286l7�4
-�*
(�%
inputs���������
� "1�.
'�$
tensor_0����������
� �
'__inference_cls_res2_layer_call_fn_4273a7�4
-�*
(�%
inputs���������
� "&�#
unknown�����������
B__inference_cls_res3_layer_call_and_return_conditional_losses_4304l7�4
-�*
(�%
inputs���������
� "1�.
'�$
tensor_0����������
� �
'__inference_cls_res3_layer_call_fn_4291a7�4
-�*
(�%
inputs���������
� "&�#
unknown�����������
B__inference_conv2_c1_layer_call_and_return_conditional_losses_4479y��9�6
/�,
*�'
inputs�����������
� "6�3
,�)
tensor_0�����������
� �
'__inference_conv2_c1_layer_call_fn_4466n��9�6
/�,
*�'
inputs�����������
� "+�(
unknown������������
B__inference_conv2_c2_layer_call_and_return_conditional_losses_4573u��7�4
-�*
(�%
inputs���������@@
� "4�1
*�'
tensor_0���������@@
� �
'__inference_conv2_c2_layer_call_fn_4560j��7�4
-�*
(�%
inputs���������@@
� ")�&
unknown���������@@�
B__inference_conv2_c3_layer_call_and_return_conditional_losses_4667u��7�4
-�*
(�%
inputs���������  
� "4�1
*�'
tensor_0���������   
� �
'__inference_conv2_c3_layer_call_fn_4654j��7�4
-�*
(�%
inputs���������  
� ")�&
unknown���������   �
B__inference_conv2_c4_layer_call_and_return_conditional_losses_4761u��7�4
-�*
(�%
inputs��������� 
� "4�1
*�'
tensor_0��������� 
� �
'__inference_conv2_c4_layer_call_fn_4748j��7�4
-�*
(�%
inputs��������� 
� ")�&
unknown��������� �
B__inference_conv2_c5_layer_call_and_return_conditional_losses_4855u��7�4
-�*
(�%
inputs��������� 
� "4�1
*�'
tensor_0���������
� �
'__inference_conv2_c5_layer_call_fn_4842j��7�4
-�*
(�%
inputs��������� 
� ")�&
unknown����������
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4645�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
.__inference_max_pooling2d_1_layer_call_fn_4640�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_4739�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
.__inference_max_pooling2d_2_layer_call_fn_4734�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_4833�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
.__inference_max_pooling2d_3_layer_call_fn_4828�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4551�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
,__inference_max_pooling2d_layer_call_fn_4546�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
?__inference_model_layer_call_and_return_conditional_losses_1880�4��������������������������D�A
:�7
-�*
	input_map�����������
p 

 
� "���
���
,�)

tensor_0_0���������   
,�)

tensor_0_1��������� 
,�)

tensor_0_2���������
� �
?__inference_model_layer_call_and_return_conditional_losses_1952�4��������������������������D�A
:�7
-�*
	input_map�����������
p

 
� "���
���
,�)

tensor_0_0���������   
,�)

tensor_0_1��������� 
,�)

tensor_0_2���������
� �
?__inference_model_layer_call_and_return_conditional_losses_4012�4��������������������������A�>
7�4
*�'
inputs�����������
p 

 
� "���
���
,�)

tensor_0_0���������   
,�)

tensor_0_1��������� 
,�)

tensor_0_2���������
� �
?__inference_model_layer_call_and_return_conditional_losses_4124�4��������������������������A�>
7�4
*�'
inputs�����������
p

 
� "���
���
,�)

tensor_0_0���������   
,�)

tensor_0_1��������� 
,�)

tensor_0_2���������
� �
$__inference_model_layer_call_fn_1503�4��������������������������D�A
:�7
-�*
	input_map�����������
p 

 
� "���
*�'
tensor_0���������   
*�'
tensor_1��������� 
*�'
tensor_2����������
$__inference_model_layer_call_fn_1808�4��������������������������D�A
:�7
-�*
	input_map�����������
p

 
� "���
*�'
tensor_0���������   
*�'
tensor_1��������� 
*�'
tensor_2����������
$__inference_model_layer_call_fn_3839�4��������������������������A�>
7�4
*�'
inputs�����������
p 

 
� "���
*�'
tensor_0���������   
*�'
tensor_1��������� 
*�'
tensor_2����������
$__inference_model_layer_call_fn_3900�4��������������������������A�>
7�4
*�'
inputs�����������
p

 
� "���
*�'
tensor_0���������   
*�'
tensor_1��������� 
*�'
tensor_2����������
>__inference_off1_layer_call_and_return_conditional_losses_4208sTU7�4
-�*
(�%
inputs���������   
� "4�1
*�'
tensor_0���������  
� �
#__inference_off1_layer_call_fn_4196hTU7�4
-�*
(�%
inputs���������   
� ")�&
unknown���������  �
>__inference_off2_layer_call_and_return_conditional_losses_4229s]^7�4
-�*
(�%
inputs��������� 
� "4�1
*�'
tensor_0���������
� �
#__inference_off2_layer_call_fn_4217h]^7�4
-�*
(�%
inputs��������� 
� ")�&
unknown����������
>__inference_off3_layer_call_and_return_conditional_losses_4250sfg7�4
-�*
(�%
inputs���������
� "4�1
*�'
tensor_0���������
� �
#__inference_off3_layer_call_fn_4238hfg7�4
-�*
(�%
inputs���������
� ")�&
unknown����������
B__inference_off_cat1_layer_call_and_return_conditional_losses_4401�d�a
Z�W
U�R
'�$
inputs_0���������� 
'�$
inputs_1���������� 
� "1�.
'�$
tensor_0���������� 
� �
'__inference_off_cat1_layer_call_fn_4394�d�a
Z�W
U�R
'�$
inputs_0���������� 
'�$
inputs_1���������� 
� "&�#
unknown���������� �
B__inference_off_cat2_layer_call_and_return_conditional_losses_4414�d�a
Z�W
U�R
'�$
inputs_0����������
'�$
inputs_1����������
� "1�.
'�$
tensor_0����������
� �
'__inference_off_cat2_layer_call_fn_4407�d�a
Z�W
U�R
'�$
inputs_0����������
'�$
inputs_1����������
� "&�#
unknown�����������
B__inference_off_cat3_layer_call_and_return_conditional_losses_4427�d�a
Z�W
U�R
'�$
inputs_0����������
'�$
inputs_1����������
� "1�.
'�$
tensor_0����������
� �
'__inference_off_cat3_layer_call_fn_4420�d�a
Z�W
U�R
'�$
inputs_0����������
'�$
inputs_1����������
� "&�#
unknown�����������
B__inference_off_res1_layer_call_and_return_conditional_losses_4322l7�4
-�*
(�%
inputs���������  
� "1�.
'�$
tensor_0���������� 
� �
'__inference_off_res1_layer_call_fn_4309a7�4
-�*
(�%
inputs���������  
� "&�#
unknown���������� �
B__inference_off_res2_layer_call_and_return_conditional_losses_4340l7�4
-�*
(�%
inputs���������
� "1�.
'�$
tensor_0����������
� �
'__inference_off_res2_layer_call_fn_4327a7�4
-�*
(�%
inputs���������
� "&�#
unknown�����������
B__inference_off_res3_layer_call_and_return_conditional_losses_4358l7�4
-�*
(�%
inputs���������
� "1�.
'�$
tensor_0����������
� �
'__inference_off_res3_layer_call_fn_4345a7�4
-�*
(�%
inputs���������
� "&�#
unknown�����������
A__inference_offsets_layer_call_and_return_conditional_losses_4457����
���
~�{
'�$
inputs_0���������� 
'�$
inputs_1����������
'�$
inputs_2����������
� "1�.
'�$
tensor_0����������*
� �
&__inference_offsets_layer_call_fn_4449����
���
~�{
'�$
inputs_0���������� 
'�$
inputs_1����������
'�$
inputs_2����������
� "&�#
unknown����������*�
"__inference_signature_wrapper_3160�@��������������������������fg]^TUKLBC9:E�B
� 
;�8
6
input_1+�(
input_1�����������"i�f
1
classes&�#
classes����������*
1
offsets&�#
offsets����������*�
B__inference_ssd_head_layer_call_and_return_conditional_losses_2971�@��������������������������fg]^TUKLBC9:B�?
8�5
+�(
input_1�����������
p 

 
� "c�`
Y�V
)�&

tensor_0_0����������*
)�&

tensor_0_1����������*
� �
B__inference_ssd_head_layer_call_and_return_conditional_losses_3075�@��������������������������fg]^TUKLBC9:B�?
8�5
+�(
input_1�����������
p

 
� "c�`
Y�V
)�&

tensor_0_0����������*
)�&

tensor_0_1����������*
� �
B__inference_ssd_head_layer_call_and_return_conditional_losses_3552�@��������������������������fg]^TUKLBC9:A�>
7�4
*�'
inputs�����������
p 

 
� "c�`
Y�V
)�&

tensor_0_0����������*
)�&

tensor_0_1����������*
� �
B__inference_ssd_head_layer_call_and_return_conditional_losses_3778�@��������������������������fg]^TUKLBC9:A�>
7�4
*�'
inputs�����������
p

 
� "c�`
Y�V
)�&

tensor_0_0����������*
)�&

tensor_0_1����������*
� �
'__inference_ssd_head_layer_call_fn_2363�@��������������������������fg]^TUKLBC9:B�?
8�5
+�(
input_1�����������
p 

 
� "U�R
'�$
tensor_0����������*
'�$
tensor_1����������*�
'__inference_ssd_head_layer_call_fn_2867�@��������������������������fg]^TUKLBC9:B�?
8�5
+�(
input_1�����������
p

 
� "U�R
'�$
tensor_0����������*
'�$
tensor_1����������*�
'__inference_ssd_head_layer_call_fn_3243�@��������������������������fg]^TUKLBC9:A�>
7�4
*�'
inputs�����������
p 

 
� "U�R
'�$
tensor_0����������*
'�$
tensor_1����������*�
'__inference_ssd_head_layer_call_fn_3326�@��������������������������fg]^TUKLBC9:A�>
7�4
*�'
inputs�����������
p

 
� "U�R
'�$
tensor_0����������*
'�$
tensor_1����������*