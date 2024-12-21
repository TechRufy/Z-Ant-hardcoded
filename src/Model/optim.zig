const std = @import("std");
const tensor = @import("tensor");
const layer = @import("layer");
const Model = @import("model");
const TensorMathError = @import("errorHandler").TensorMathError;
const DenseLayer = @import("denselayer");

pub const Optimizers = enum {
    SGD,
    Adam,
    RMSprop,
};

// Define the Optimizer struct with the optimizer function, learning rate, and allocator
pub fn Optimizer(comptime T: type, func: fn (comptime type, f64, *const std.mem.Allocator) type, lr: f64, allocator: *const std.mem.Allocator) type {
    const optim = func(T, lr, allocator){};
    return struct {
        optimizer: func(T, lr, allocator) = optim, // Instantiation of the optimizer (e.g., SGD, Adam)

        pub fn step(self: *@This(), model: *Model.Model(T, allocator)) !void {
            // Directly call the optimizer's step function
            try self.optimizer.step(model);
        }
    };
}

// Define the SGD optimizer
// NEED TO BE MODIFIED IF NEW LAYERS ARE ADDED
pub fn optimizer_SGD(T: type, lr: f64, allocator: *const std.mem.Allocator) type {
    return struct {
        learning_rate: f64 = lr,
        allocator: *const std.mem.Allocator = allocator,

        // Step function to update weights and biases using gradients
        pub fn step(self: *@This(), model: *Model.Model(T, allocator)) !void {
            var counter: u32 = 0;
            for (model.layers.items) |layer_| {
                switch (layer_.layer_type) {
                    .DenseLayer => { // need to put the type of layer as input, now only work for dense layer
                        const myDense: *DenseLayer.DenseLayer(T, allocator) = @ptrCast(@alignCast(layer_.layer_ptr));
                        const weight_gradients = &myDense.w_gradients;
                        const bias_gradients = &myDense.b_gradients;
                        const weight = &myDense.weights;
                        const bias = &myDense.bias;

                        //need to talk with the guys about this class, probably need a big rework of the layer structure

                        //std.debug.print("\n ------ step {}", .{counter});

                        try self.update_tensor(weight, weight_gradients);
                        try self.update_tensor(bias, bias_gradients);
                    },
                    else => {},
                }
                counter += 1;
            }
        }

        // Helper function to update tensors
        fn update_tensor(self: *@This(), t: *tensor.Tensor(T), gradients: *tensor.Tensor(T)) !void {
            if (t.size != gradients.size) return TensorMathError.InputTensorDifferentSize;
            //we move in the opposite direction of the gradient
            for (t.data, 0..) |*value, i| {
                value.* -= gradients.data[i] * self.learning_rate;
            }
        }
    };
}

pub fn optimizer_ADAM(T: type, lr: f64, allocator: *const std.mem.Allocator) type {
    return struct {
        learning_rate: f64 = lr,
        allocator: *const std.mem.Allocator = allocator,
        alpha: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        m: f64,
        v: f64,

        pub fn init(self: *@This(), alpha: f64, beta1: f64, beta2: f64, epsilon: f64) !void {
            self.alpha = alpha;
            self.beta1 = beta1;
            self.beta2 = beta2;
            self.epsilon = epsilon;
        }

        // Step function to update weights and biases using gradients
        pub fn step(self: *@This(), model: *Model.Model(T, allocator)) !void {
            var counter: u32 = 0;
            for (model.layers.items) |layer_| {
                switch (layer_.layer_type) {
                    .DenseLayer => { // need to put the type of layer as input, now only work for dense layer
                        const myDense: *DenseLayer.DenseLayer(T, allocator) = @ptrCast(@alignCast(layer_.layer_ptr));
                        const weight_gradients = &myDense.w_gradients;
                        const bias_gradients = &myDense.b_gradients;
                        const weight = &myDense.weights;
                        const bias = &myDense.bias;

                        //need to talk with the guys about this class, probably need a big rework of the layer structure

                        //std.debug.print("\n ------ step {}", .{counter});

                        try self.update_tensor(weight, weight_gradients);
                        try self.update_tensor(bias, bias_gradients);
                    },
                    else => {},
                }
                counter += 1;
            }
        }

        // Helper function to update tensors
        fn update_tensor(self: *@This(), t: *tensor.Tensor(T), gradients: *tensor.Tensor(T)) !void {
            if (t.size != gradients.size) return TensorMathError.InputTensorDifferentSize;

            for (t.data, 0..) |*value, i| {
                self.m = self.m * self.beta1 + (1 - self.beta1) * gradients.data[i];
                //i need the second gradient to continue
                value.* -= gradients.data[i] * self.learning_rate;
            }
        }
    };
}
