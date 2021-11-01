import numpy as np

class Convolution:
    def __init__(self, nb_filters, filter_size):
        self.nb_filters = nb_filters
        self.filter_size = filter_size
        self.conv_filter = np.random.randn(nb_filters, filter_size, filter_size) / (filter_size * filter_size)
        
    def get_image_patchs(self, image):
        height, width = image.shape
        self.image = image
        for i in range(height - self.filter_size + 1):
            for j in range(width - self.filter_size + 1):
                image_patch = image[i : (i + self.filter_size), j: (j + self.filter_size)]
                yield image_patch, i, j
                
    def forward_propagation(self, image):
        height, width = image.shape
        conv_out = np.zeros((height - self.filter_size + 1, width - self.filter_size + 1, self.nb_filters))
        for image_patch, i, j in self.get_image_patchs(image):
            conv_out[i,j] = np.sum(image_patch * self.conv_filter, axis=(1,2))
        return conv_out
    
    def backward_propagation(self, dL_dout, learning_rate):
        dL_dF_params = np.zeros(self.conv_filter.shape)
        for image_patch, i, j in self.get_image_patchs(self.image):
            for k in range(self.nb_filters):
                dL_dF_params[k] += image_patch * dL_dout[i, j, k]
        #update params
        self.conv_filter -= learning_rate * dL_dF_params
        return dL_dF_params

class MaxPool:
    def  __init__(self, filter_size):
        self.filter_size = filter_size
    
    def get_image_patchs(self, image):
        new_height = image.shape[0] // self.filter_size
        new_width = image.shape[1] // self.filter_size
        self.image = image
        for i in range(new_height):
            for j in range(new_width):
                image_patch = image[i * self.filter_size : i * self.filter_size + self.filter_size, j * self.filter_size : (j * self.filter_size + self.filter_size)]
                yield image_patch, i, j
                
    def forward_propagation(self, image):
        height, width, nb_filters = image.shape
        max_pool_out = np.zeros((height // self.filter_size, width // self.filter_size, nb_filters))
        for image_patch, i, j in self.get_image_patchs(image):
            max_pool_out[i,j] = np.amax(image_patch, axis=(0,1))
        return max_pool_out
    
    def backward_propagation(self, dL_dout):
        dL_dmax_pool = np.zeros(self.image.shape)
        for image_patch, i, j in self.get_image_patchs(self.image):
            height, width, nb_filters = image_patch.shape
            max_val = np.amax(image_patch, axis=(0,1))
            for i1 in range(height):
                for j1 in range(width):
                    for k1 in range(nb_filters):
                        if image_patch[i1, j1, k1] == max_val[k1]:
                            dL_dmax_pool[i * self.filter_size + i1, j * self.filter_size + j1, k1] = dL_dout[i,j,k1]
        return dL_dmax_pool

class SoftMax:
    def __init__(self, input_node, softmax_node):
        self.weight = np.random.randn(input_node, softmax_node) / input_node
        self.bias = np.zeros(softmax_node)
    
    def forward_propagation(self, image):
        self.orig_shape = image.shape
        
        image_flatten = image.flatten()
        self.image_flatten = image_flatten
        output_val = np.dot(image_flatten, self.weight) + self.bias
        self.out = output_val
        exp_out = np.exp(output_val)
        return  exp_out / np.sum(exp_out, axis=0)
    
    def backward_propagation(self, dL_dout, learning_rate):
        for i, gradient in enumerate(dL_dout):
            if gradient == 0:
                continue
            transform_equa = np.exp(self.out)
            s_total = np.sum(transform_equa)
            #Gradients with respect to out (z)
            dy_dz = -transform_equa[i] * transform_equa / (s_total ** 2) 
            dy_dz[i] = transform_equa[i] * (s_total - transform_equa[i]) / (s_total ** 2)
            #Gradients od totals against weights /bias/input
            dz_dw = self.image_flatten
            dz_db = 1
            dz_d_inp = self.weight
            #Gradients of loss against totals
            dL_dz = gradient * dy_dz
            #Gradients of loss against weight/biases/input
            dL_dw = dz_dw[np.newaxis].T @ dL_dz[np.newaxis]
            dL_db = dL_dz * dz_db
            dL_d_inp = dz_d_inp @ dL_dz
        #Update weights and biases
        self.weight -= learning_rate * dL_dw
        self.bias -= learning_rate * dL_db
        return dL_d_inp.reshape(self.orig_shape)
    
class CNN:
    def __init__(self, image_size, nb_class, nb_channel, conv_filter_size, pool_filter_size):
        self.nb_class = nb_class
        self.conv = Convolution(nb_channel, conv_filter_size)
        self.maxpool = MaxPool(pool_filter_size)
        self.softmax = SoftMax((((image_size - conv_filter_size +1) // pool_filter_size)**2)  * nb_channel, nb_class)
        
    def CNN_forward_propa(self, image, label):
        out = self.conv.forward_propagation(image/255)
        out = self.maxpool.forward_propagation(out)
        out = self.softmax.forward_propagation(out)
        #Calculate cross-entropy loss accuracy
        cross_entropy_loss = - np.log(out[label])
        accuracy_eval = 1 if np.argmax(out) == label else 0
        return out, cross_entropy_loss, accuracy_eval

    def CNN_backward_propa(self, image, label, learning_rate=0.005):
        out, loss, acc = self.CNN_forward_propa(image, label)
        #Calculate initial gradient
        gradient = np.zeros(self.nb_class)
        gradient[label] = -1 / out[label]
        #Backward
        grad = self.softmax.backward_propagation(gradient, learning_rate)
        grad = self.maxpool.backward_propagation(grad)
        grad = self.conv.backward_propagation(grad, learning_rate)
        return loss, acc 
    
    def training(self, epoch):
        for epoch in range(epoch):
            print("Epoch ", epoch+1, "---->")
            loss = 0
            num_correct = 0
            for i, (img, label) in enumerate(zip(X_train, y_train)):
                if i % 100 == 0:
                    print(i+1, " steps out of 100 steps: Average Loss ",loss/100," and Accuracy: ", num_correct,"%")
                    loss = 0
                    num_correct = 0
                l, acc = self.CNN_backward_propa(img, label)
                loss += l
                num_correct += acc
                
    def score(self, X_test, y_test):
        loss = 0
        num_correct = 0
        for i, (img, label) in enumerate(zip(X_test, y_test)):
            l, acc = self.CNN_backward_propa(img, label)
            loss += l
            num_correct += acc
        nb_sample = len(y_test)
        return loss / nb_sample, num_correct / nb_sample
            
    def predict(self, images):
        result = []
        for X in images:
            out = self.conv.forward_propagation(X/255)
            out = self.maxpool.forward_propagation(out)
            out = self.softmax.forward_propagation(out)
            result.append(np.argmax(out))
        return result
        
