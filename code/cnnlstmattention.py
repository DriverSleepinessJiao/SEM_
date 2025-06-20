from __future__ import print_function
import numpy as np
import scipy.io as sio
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report, confusion_matrix
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Reshape, Input, GlobalAveragePooling1D
from keras.layers import Conv1D, MaxPooling1D, LSTM, BatchNormalization, Concatenate
from keras_self_attention import SeqSelfAttention
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras import backend as K
from sklearn.utils import class_weight
from keras.regularizers import l2
from keras.layers import Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau



# ---------------------------测试集---------------------------
Testalphafile1 = r"pos_Test_x.mat"
TestData1 = sio.loadmat(Testalphafile1)['Test_x']
TestAlpha1 = TestData1[:, :, 0]
TestSEM1 = TestData1[:, :, 1]

Testalphafile0 = r"neg_2891.mat"
TestData0 = sio.loadmat(Testalphafile0)['remaining_data']
TestAlpha0 = TestData0[:, :, 0]
TestSEM0 = TestData0[:, :, 1]

TestAlpha = np.concatenate((TestAlpha1, TestAlpha0), axis=0)
TestAlpha = np.expand_dims(TestAlpha, axis=2)
TestSEM = np.concatenate((TestSEM1, TestSEM0), axis=0)
TestSEM = np.expand_dims(TestSEM, axis=2)
# ----------------------标签-----------------------------------------
num_classes = 2

y_test = np.concatenate((np.ones(TestAlpha1.shape[0]), np.zeros(TestAlpha0.shape[0])), axis=0)
y_test = y_test.astype("float32")
y_test = np_utils.to_categorical(y_test, num_classes)

# ----------------------标签-----------------------------------------
y_test = np.concatenate((np.ones(TestAlpha1.shape[0], ), np.zeros(TestAlpha0.shape[0], )), axis=0)  # [12000,]


# 加载数据
TrainAlphafile1 = r"pos_Train_x.mat"
TrainAlpha1 = sio.loadmat(TrainAlphafile1)
x_train1 = TrainAlpha1['Train_x']  # (1174, 500)

TrainAlphafile0 = r"neg_6746.mat"
TrainAlpha0 = sio.loadmat(TrainAlphafile0)
x_train0 = TrainAlpha0['subset_data']  # (1428, 500)

TrainAlpha1 = x_train1[:, :, 0]
TrainSEM1 = x_train1[:, :, 1]

TrainAlpha0 = x_train0[:, :, 0]
TrainSEM0 = x_train0[:, :, 1]

TrainAlpha = np.concatenate((TrainAlpha1, TrainAlpha0), axis=0)
TrainAlpha = np.expand_dims(TrainAlpha, axis=2)
TrainSEM = np.concatenate((TrainSEM1, TrainSEM0), axis=0)
TrainSEM = np.expand_dims(TrainSEM, axis=2)

# 标签
y_train = np.concatenate((np.ones(TrainAlpha1.shape[0]), np.zeros(TrainAlpha0.shape[0])), axis=0)  # [12000,]

# One-hot encoding
y_train = np_utils.to_categorical(y_train, 2)

# 计算类别权重
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(np.argmax(y_train, axis=1)),
    y=np.argmax(y_train, axis=1)
)
class_weights = {0: class_weights[0], 1: class_weights[1]}

# 定义超参数搜索空间
alpha_kernels1 = [50, 150, 250]
alpha_kernels2 = [50, 150, 250]
sem_kernels1 = [50, 150, 250]
sem_kernels2 = [50, 150, 250]
lstm_units = [50, 100, 150]     # LSTM隐藏单元数

# 创建模型
def create_model(alpha_k1, alpha_k2, sem_k1, sem_k2, lstm_us):
    inp1 = Input(shape=TrainAlpha.shape[1:])
    inp2 = Input(shape=TrainSEM.shape[1:])

    # Alpha信号处理
    model_m_alpha = Sequential()
    model_m_alpha.add(Conv1D(filters=64, kernel_size=alpha_k1, strides=1, padding='valid', data_format='channels_last', input_shape=TrainAlpha.shape[1:]))
    model_m_alpha.add(BatchNormalization())
    model_m_alpha.add(Activation('relu'))
    model_m_alpha.add(Conv1D(filters=64, kernel_size=alpha_k2, strides=1, padding='same'))
    model_m_alpha.add(BatchNormalization())
    model_m_alpha.add(Activation('relu'))
    model_m_alpha.add(MaxPooling1D(pool_size=2, strides=2))

    # model_m_alpha.add(LSTM(units=lstm_units, return_sequences=True))
    # model_m_alpha.add(SeqSelfAttention(attention_activation='tanh'))
    # model_m_alpha(GlobalAveragePooling1D())
    # model_m_alpha.add(Dropout(0.2))
    model_m_alpha.add(LSTM(units=lstm_us, return_sequences=True))  # 返回整个序列的输出
    model_m_alpha.add(SeqSelfAttention(attention_activation='tanh'))
    model_m_alpha.add(Dropout(0.2))
    model_m_alpha.add(GlobalAveragePooling1D())

    model_m_alpha_output = model_m_alpha(inp1)
    model_m_alpha.summary()  # 在这里调用 model.summary()
    # SEM信号处理
    model_m_sem = Sequential()
    model_m_sem.add(Conv1D(filters=64, kernel_size=sem_k1, strides=1, padding='valid', data_format='channels_last', input_shape=TrainSEM.shape[1:]))
    model_m_sem.add(BatchNormalization())
    model_m_sem.add(Activation('relu'))
    model_m_sem.add(Conv1D(filters=64, kernel_size=sem_k2, strides=1, padding='same'))
    model_m_sem.add(BatchNormalization())
    model_m_sem.add(Activation('relu'))
    model_m_sem.add(MaxPooling1D(pool_size=2, strides=2))

    # model_m_sem.add(LSTM(units=lstm_units, return_sequences=True))
    # model_m_sem.add(SeqSelfAttention(attention_activation='tanh'))
    # model_m_sem(GlobalAveragePooling1D())
    # model_m_sem.add(Dropout(0.2))

    model_m_sem.add(LSTM(units=lstm_us, return_sequences=True))  # 返回整个序列的输出
    model_m_sem.add(SeqSelfAttention(attention_activation='tanh'))
    model_m_sem.add(Dropout(0.2))
    model_m_sem.add(GlobalAveragePooling1D())  # 添加全局平均池化层来减少维度
    model_m_sem_output = model_m_sem(inp2)
    model_m_alpha.summary()  # 在这里调用 model.summary()
    # 合并并行处理后的结果
    mrg = Concatenate()([model_m_alpha_output, model_m_sem_output])

    # FC层
    dense = Dense(128, activation='relu')(mrg)
    dropout = Dropout(0.5)(dense)
    out = Dense(2, activation='softmax')(dropout)
    #dense = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(mrg)
    #op = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(dense)
    #out = Dense(2, activation='softmax')(op)

    # 构建模型
    model = Model(inputs=[inp1, inp2], outputs=out)
    model.compile(optimizer=Adam(learning_rate=5e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()  # 在这里调用 model.summary()
    return model

# 五折交叉验证和参数优化
best_accuracy = 0
best_f1_score = 0
best_params = {}

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 定义一个空的DataFrame来存储结果
results_df = pd.DataFrame(columns=['Alpha Kernel 1', 'Alpha Kernel 2', 'SEM Kernel 1', 'SEM Kernel 2', 'LSTM Units', 'Accuracy', 'F1 Score', 'Accuracy Std'])

for alpha_k1 in alpha_kernels1:
    for alpha_k2 in alpha_kernels2:
        for sem_k1 in sem_kernels1:
            for sem_k2 in sem_kernels2:
                for lstm_us in lstm_units:
                    print(f"Evaluating model with Alpha Kernels: {alpha_k1}, {alpha_k2}, SEM Kernels: {sem_k1}, {sem_k2}, LSTM Units: {lstm_us}")
                    fold_accuracies = []
                    fold_f1_scores = []
                    fold_confusion_matrices = []

                    for fold, (train_idx, val_idx) in enumerate(kfold.split(TrainSEM, y_train.argmax(axis=1))):
                        x_train_fold, x_val_fold = TrainAlpha[train_idx], TrainAlpha[val_idx]
                        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
                        x_train_sem_fold, x_val_sem_fold = TrainSEM[train_idx], TrainSEM[val_idx]

                        model = create_model(alpha_k1, alpha_k2, sem_k1, sem_k2, lstm_us)
                        model_save_path = f'best_model_fold{fold + 1}.h5'
                        checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
                        early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)
                        reduceLROnPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1),
                        model.fit([x_train_fold, x_train_sem_fold], y_train_fold, batch_size=128, epochs=100, validation_data=([x_val_fold, x_val_sem_fold], y_val_fold),
                                  callbacks=[checkpoint, early_stopping,reduceLROnPlateau], verbose=0, class_weight=class_weights)

                        # 评估验证集
                        val_pred = model.predict([x_val_fold, x_val_sem_fold])
                        y_val_pred = (val_pred[:, 1] > 0.5).astype(int)
                        acc = accuracy_score(y_val_fold.argmax(axis=1), y_val_pred)
                        f1 = f1_score(y_val_fold.argmax(axis=1), y_val_pred)
                        fold_accuracies.append(acc)
                        fold_f1_scores.append(f1)
                        conf_mat = confusion_matrix(y_val_fold.argmax(axis=1), y_val_pred)
                        fold_confusion_matrices.append(conf_mat)
                        print(f"Fold {fold + 1} Confusion Matrix:\n{conf_mat}")
                        print(f"Fold {fold + 1} Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
                    # 计算当前参数组合的平均验证准确率和F1分数
                    mean_accuracy = np.mean(fold_accuracies)
                    mean_f1_score = np.mean(fold_f1_scores)
                    accuracy_std = np.std(fold_accuracies)
                    print(f"Mean Validation Accuracy for Conv alpha: {alpha_k1}, {alpha_k2}, Conv sem: {sem_k1}, {sem_k2}, LSTM Units: {lstm_us} - {mean_accuracy:.4f}")
                    print(f"Mean Validation F1 Score for Conv alpha: {alpha_k1}, {alpha_k2}, Conv sem: {sem_k1}, {sem_k2}, LSTM Units: {lstm_us} - {mean_f1_score:.4f}")
                    print(f"Accuracy Std for Conv alpha: {alpha_k1}, {alpha_k2}, Conv sem: {sem_k1}, {sem_k2}, LSTM Units: {lstm_us} - {accuracy_std:.4f}")

                    # 将结果添加到DataFrame中
                    results_df = pd.concat([results_df, pd.DataFrame({
                        'Alpha Kernel 1': [alpha_k1],
                        'Alpha Kernel 2': [alpha_k2],
                        'SEM Kernel 1': [sem_k1],
                        'SEM Kernel 2': [sem_k2],
                        'LSTM Units': [lstm_us],
                        'Accuracy': [mean_accuracy],
                        'F1 Score': [mean_f1_score],
                        'Accuracy Std': [accuracy_std]
                    })], ignore_index=True)

                    # 更新最佳参数组合
                    if mean_accuracy > best_accuracy or (mean_accuracy == best_accuracy and mean_f1_score > best_f1_score):
                        best_accuracy = max(mean_accuracy, best_accuracy)
                        best_f1_score = max(mean_f1_score, best_f1_score)
                        best_params = {
                            'Alpha Kernel 1': alpha_k1,
                            'Alpha Kernel 2': alpha_k2,
                            'SEM Kernel 1': sem_k1,
                            'SEM Kernel 2': sem_k2,
                            'LSTM Units': lstm_us
                        }
# 将DataFrame保存为Excel文件
results_df.to_excel('dzCNNLSTMATTENTION_parameter_optimization_results.xlsx', index=False)

print(f"Best Params: {best_params} with Validation Accuracy: {best_accuracy:.4f} and F1 Score: {best_f1_score:.4f}")


y_test = np.concatenate((np.ones(TestSEM1.shape[0], ), np.zeros(TestSEM0.shape[0], )), axis=0)  # [12000,]


# 测试集评估
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# 使用最佳参数重新训练模型并进行测试
best_model = create_model(best_params['Alpha Kernel 1'], best_params['Alpha Kernel 2'],
                          best_params['SEM Kernel 1'], best_params['SEM Kernel 2'],
                          best_params['LSTM Units'])

# # 在全量训练数据上训练
# best_model.fit([TrainAlpha, TrainSEM], y_train, batch_size=128, epochs=100, validation_split=0.2,
#                callbacks=[EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)], class_weight=class_weights)
# 定义回调函数
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)
reduceLROnPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)

# 训练模型
best_model.fit([TrainAlpha, TrainSEM], y_train, batch_size=128, epochs=100, validation_split=0.2,
               callbacks=[early_stopping, reduceLROnPlateau], verbose=0, class_weight=class_weights)
# 预测测试集
test_pred = best_model.predict([TestAlpha, TestSEM])

# 将预测结果转为类别标签
y_test_pred = (test_pred[:, 1] > 0.5).astype(int)

# 计算准确率和F1分数
# 计算准确率和F1分数
accuracy = accuracy_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
conf_mat = confusion_matrix(y_test, y_test_pred)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Test Confusion Matrix:\n{conf_mat}")

# 将结果保存为Excel文件
test_results = pd.DataFrame({
    'Alpha Kernel 1': [best_params['Alpha Kernel 1']],
    'Alpha Kernel 2': [best_params['Alpha Kernel 2']],
    'SEM Kernel 1': [best_params['SEM Kernel 1']],
    'SEM Kernel 2': [best_params['SEM Kernel 2']],
    'LSTM Units': [best_params['LSTM Units']],
    'Test Accuracy': [accuracy],
    'Test F1 Score': [f1],
    'Test Confusion Matrix': [conf_mat.tolist()]
})

# 保存最佳结果到Excel
with pd.ExcelWriter('CNNLSTMATTENTION_parameter_optimization_results.xlsx', mode='a', engine='openpyxl') as writer:
    test_results.to_excel(writer, sheet_name='Test Results', index=False)

print(f"Test results saved to 'CNNLSTMATTENTION_parameter_optimization_results.xlsx'.")

