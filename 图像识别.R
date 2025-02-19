# 安装 tensorflow 和 keras 包
install.packages("keras")
install.packages("tensorflow")

# 加载 keras 和 tensorflow 库
library(keras)
library(tensorflow)

# 安装 keras 和 tensorflow 后端支持（会自动下载并安装）
install_keras()
# 加载 CIFAR-10 数据集
dataset <- dataset_cifar10()

# 提取训练数据和测试数据
x_train <- dataset$train$x
y_train <- dataset$train$y
x_test <- dataset$test$x
y_test <- dataset$test$y

# 数据预处理：将像素值缩放到 0-1 范围
x_train <- x_train / 255
x_test <- x_test / 255
# 构建 CNN 模型
model <- keras_model_sequential() %>%
  # 第一层卷积层，使用 32 个 3x3 的卷积核
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(32, 32, 3)) %>%
  # 最大池化层
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  # 第二层卷积层
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  # 最大池化层
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  # 扁平化层，将多维输入一维化
  layer_flatten() %>%
  # 全连接层
  layer_dense(units = 128, activation = 'relu') %>%
  # 输出层，10 类分类，使用 softmax 激活函数
  layer_dense(units = 10, activation = 'softmax')

# 查看模型结构
summary(model)
model %>% compile(
  loss = 'sparse_categorical_crossentropy',  # 对于分类任务使用交叉熵损失函数
  optimizer = optimizer_adam(),              # 使用 Adam 优化器
  metrics = c('accuracy')                    # 使用准确率作为评估指标
)
history <- model %>% fit(
  x_train, y_train, 
  epochs = 10,          # 设置训练周期数
  batch_size = 64,      # 每次训练的样本数量
  validation_data = list(x_test, y_test)  # 使用测试数据进行验证
)
# 在测试集上评估模型
score <- model %>% evaluate(x_test, y_test)
cat('Test loss:', score$loss, "\n")
cat('Test accuracy:', score$accuracy, "\n")
# 加载图片并进行预处理
img_path <- "path/to/your/test/image.png"  # 替换为你的图片路径
img <- image_load(img_path, target_size = c(32, 32))
img_array <- image_to_array(img)
img_array <- array_reshape(img_array, c(1, 32, 32, 3))  # 形状调整为 (1, 32, 32, 3)
img_array <- img_array / 255  # 归一化

# 进行预测
predictions <- model %>% predict(img_array)

# 输出预测结果
predicted_class <- which.max(predictions) - 1  # 类别从0开始
cat("Predicted class:", predicted_class, "\n")
