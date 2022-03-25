open Tensorflow
open Tensorflow.Keras.Layers
open type Tensorflow.Binding
open type Tensorflow.KerasApi


[<EntryPoint>]
let main argv =

    // Load the handwritten digit dataset directly.
    // It is already split into training and test data for us.
    let (image_train, label_train, image_test, label_test) = 
        keras.datasets.mnist.load_data().Deconstruct()

    // The training images are 24 * 24 pixels, and there are 60,000 of them.
    // We need to reshape them into a single row of 784 pixels.
    // We also need to scale them from a range of 0-255 to a range of 0-1.
    let image_train_flat = 
        (image_train.reshape(Shape [| 60000; 784 |])) / 255f

    // Same here for the test images, although there are 10,000 of them.
    let image_test_flat = 
        (image_test.reshape(Shape [| 10000; 784 |])) / 255f

    // Here we define our model input shape
    let inputs = new Tensors([|keras.Input(Shape [| 784 |])|])
    
    let layers = LayersApi()

    // We are going to have a single layer of processing.
    // This will divide our inputs into one of ten groups.
    let outputs = 
         inputs
         // You could insert many layers here, for example
         // |> layers.Dense(1024, activation = keras.activations.Relu).Apply
         |> layers.Dense(10).Apply

    // This is an alternative way to flatten and normalise the data as part of the model pipeline. 
    // If you took this approach you wouldn't need the reshape commands, you could use the unflattened data directly.
    //let outputs = 
    //    inputs
    //    |> layers.Flatten().Apply
    //    |> layers.Rescaling(1.0f / 255f).Apply
    //    |> layers.Dense(10).Apply
    
    // We create our model by combining the inputs and outputs.
    let model = keras.Model(inputs, outputs, name = "Digit Recognition")

    // This will print a nice summary of the model structure to the console
    model.summary() 

    // Here we decide which optimiser and cost functions we wish to use
    model.compile(
        // Schochastic Gradient Descent with a learning rate (alpha) of 0.1
        // https://keras.io/api/optimizers/
        keras.optimizers.SGD(0.1f), 
        // Categorisation specific loss function, with a flag to enable logistic normalisation.
        // https://keras.io/api/losses/probabilistic_losses/
        keras.losses.SparseCategoricalCrossentropy(from_logits = true),
        metrics = [| "accuracy" |])

    // Train the model
    model.fit(
        image_train_flat.numpy(), // .numpy() converts to a Python NDArray.
        label_train,
        epochs = 1) // One pass over the data set

    // Check the model accuracy using the test data.
    model.evaluate(image_test_flat.numpy(), label_test, verbose = 2)

    0
    