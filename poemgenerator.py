# Import packages
import numpy as np
import random
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import LSTM, CuDNNLSTM, Dropout, Dense, Activation, Embedding
from keras.optimizers import Adam
from language_utils import greek_to_greeklish, greeklish_to_greek
from keras.utils import plot_model
import pickle


def load_dataset(input_file='dataset/full_sonnets.txt'):
    # Load the dataset (already converted to greeklish)
    with open(input_file, 'r', encoding='utf-8-sig') as file:
        dataset = file.read()
    # Make upper to lower case for simplicity
    dataset = dataset.lower()
    return dataset


def create_training_splits(dataset):
    # Store each unique character that appears in the dataset in a sorted list
    unique_characters = sorted(list(set(dataset)))
    print(unique_characters)
    print(len(unique_characters))
    # Creating dictionary to assign a number to each character
    number_to_character = {n: c for n, c in enumerate(unique_characters)}
    # Creating dictionary with the assigned character to each number
    character_to_number = {c: n for n, c in enumerate(unique_characters)}

    # Create list to store train and target sequences
    sequence = []
    target_character = []

    # Overall length of the whole text
    dataset_length = len(dataset)

    # Length of the sentence we will use to train the model
    sequence_length = 10
    # Step indicates the frame of characters that our model learns
    step = 1

    # Create sequences of characters and predict the next character
    for i in range(0, dataset_length - sequence_length, step):
        train_sequence = dataset[i:i + sequence_length]
        target = dataset[i + sequence_length]
        sequence.append([character_to_number[c] for c in train_sequence])
        target_character.append(character_to_number[target])

    # Convert and rescale input into 3-dimensional shape (input of LSTM model)
    x = np.reshape(sequence, (len(sequence), sequence_length))

    # Convert predicted character into categorical values (0,1)
    y = np_utils.to_categorical(target_character)
    return x, y, unique_characters, character_to_number, number_to_character



def create_model(input_len=10, n_characters= 53):
    # Time to create our model
    model = Sequential()
    # Add an LSTM layer with 500 units
    model.add(Embedding(n_characters, 64, input_length=input_len))
    model.add(CuDNNLSTM(500, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(CuDNNLSTM(500, return_sequences=True))
    model.add(Dropout(0.2))
    # Add a dropout layer to avoid overfitting
    # model.add(Dropout(0.2))
    model.add((CuDNNLSTM(500)))
    model.add(Dropout(0.2))
    # Add a fully connected dense output layer
    model.add(Dense(n_characters))
    # Set activation function
    model.add(Activation('softmax'))

    return model


def train_model(input_file='dataset/full_sonnets.txt', output_model='models/non_accented/3LSTM500.h5', dictionary_path='dictionary.pickle'):
    dataset = load_dataset(input_file)

    x_train, y_train, unique_characters, character_to_number, number_to_character = create_training_splits(dataset)

    print(x_train.shape)
    print(y_train.shape)
    # Save the character to id mapping
    with open(dictionary_path, "wb") as f:
        pickle.dump(number_to_character, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(character_to_number, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Create model
    model = create_model(x_train.shape[1], y_train.shape[1])

    # Set loss function, optimizer and learning rate
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001))

    model.load_weights(output_model)
    # Train the model
    model.fit(x_train, y_train, epochs=1, batch_size=50, verbose=1)
    plot_model(model, to_file='pics/baseline_model.png', show_layer_names=True,show_shapes=True)

    # Save predictions to output model
    model.save(output_model)

# Roullete Wheel Selection
def select_probability(predictions):
    rand = random.random()
    total = 0.0
    for i in range(len(predictions)):
        total += predictions[i]
        if total > rand:
            return i


def generate_poem(dictionary_path='dictionary.pickle', model_path='models/non_accented/3LSTM500.h5'):

    # Save the character to id mapping
    with open(dictionary_path, "rb") as f:
        number_to_character = pickle.load(f)
        character_to_number = pickle.load(f)
    sequence_length = 10

    # Load model
    model = create_model(sequence_length, len(number_to_character))
    model.load_weights(model_path)

    # Ask user to define the first characters of the poem
    start_input = input("Write the first characters of the poem in greek, please: ").lower()
    while (len(start_input) < sequence_length):
        start_input = input("THe sequence of characters is short. Please try again: ").lower()
    greeklish_input = greek_to_greeklish(start_input)
    list_input = list(greeklish_input)
    start_sentence = list_input[0:sequence_length]

    # Convert the previous sentence from characters to values using dictionary
    converted_sentence = [character_to_number[c] for c in start_sentence if c in character_to_number]

    patience = 0
    # Define the number of characters which will be selected random for each word
    roulette_patience = 3

    # Make Predictions
    for i in range(400):
        # Reshape and rescale sentence with into the 3 dimension shape as before
        x = np.reshape(converted_sentence, (1, len(converted_sentence)))
        # Take model predictions
        model_predictions = model.predict(x)[0]
        # For the pre define length of characters chose random the next prediction
        if patience < roulette_patience:
            prediction = select_probability(model_predictions)
        else:
            # Chose the prediction with the highest probability
            prediction = np.argmax(model_predictions)

        patience+=1
        # Do not increase counter if next character is \n or space
        if prediction == 0 or prediction == 1:
            patience = 0
        # Convert values to characters and append them to the initial sentence
        start_sentence.append(number_to_character[prediction])
        # Update start sentence with the predicted character
        converted_sentence.append(prediction)
        # Remove first character of the sentence
        converted_sentence = converted_sentence[1:len(converted_sentence)]

    # Print the final generated poem
    generated_poem = ""
    for character in start_sentence:
        generated_poem = generated_poem + character
    return generated_poem


# train_model()

greeklish_poem = generate_poem()
greek_poem = greeklish_to_greek(greeklish_poem)
print(greek_poem)
