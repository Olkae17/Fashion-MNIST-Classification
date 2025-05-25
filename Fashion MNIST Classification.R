# Loading necessary libraries
library(keras)
library(ggplot2)
library(tidyr)
library(tensorflow)

# Loading Fashion MNIST dataset
fashion_mnist <- dataset_fashion_mnist()
c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test

# Preprocessing the data
# Normalizing pixel values to be between 0 and 1
train_images <- train_images / 255
test_images <- test_images / 255

# Reshaping the data to include channel dimension
train_images <- array_reshape(train_images, c(nrow(train_images), 28, 28, 1))
test_images <- array_reshape(test_images, c(nrow(test_images), 28, 28, 1))

# Converting labels
train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

# Defining the 6-layer CNN model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu", 
                input_shape = c(28,28,1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  layer_flatten() %>%
  
  layer_dense(units = 128, activation = "relu") %>%
  
  layer_dense(units = 10, activation = "softmax")

# Compiling the model
model %>% compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

# Training the model
history <- model %>% fit(
  train_images, train_labels,
  epochs = 15,
  batch_size = 128,
  validation_split = 0.1,
  verbose = 2
)

# Evaluating the model
score <- model %>% evaluate(test_images, test_labels, verbose = 0)
cat('Test loss:', score[1], '\n')
cat('Test accuracy:', score[2], '\n')

# Making predictions
class_names = c('T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat', 
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot')

# Selecting 10 test images to display
test_images_display <- test_images[1:10,,,]
predictions <- model %>% predict(test_images_display)

# Displaying images with predictions
par(mfrow=c(2,5), mar=c(0, 0, 1, 0))
for (i in 1:10) {
  img <- test_images_display[i,,,1]
  img <- t(apply(img, 2, rev))
  predicted_label <- which.max(predictions[i,]) - 1
  true_label <- which.max(test_labels[i,]) - 1
  plot(1, type="n", xlim=c(0,28), ylim=c(0,28), xlab="", ylab="", axes=FALSE)
  rasterImage(img, 0, 0, 28, 28)
  title(main=paste(class_names[predicted_label + 1], "\n(true: ", class_names[true_label + 1], ")", sep=""),
        col.main=ifelse(predicted_label == true_label, "blue", "red"))
}

# Saving the model
save_model_hdf5(model, "fashion_mnist_model.h5")