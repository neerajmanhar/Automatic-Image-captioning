import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os
from torchvision import transforms
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pyttsx3
from gtts import gTTS
from googletrans import Translator

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize([224, 224], Image.LANCZOS)
        
        if transform is not None:
            image = transform(image).unsqueeze(0)
        
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def text_to_speech_gtts_auto(text, output_file="output.mp3"):
    try:
        # Generate speech audio
        tts = gTTS(text=text, lang='hi')
        tts.save(output_file)
        
        # Automatically play the audio
        if os.name == 'nt':  # Windows
            os.system(f"start {output_file}")
        elif os.uname().sysname == 'Darwin':  # macOS
            os.system(f"open {output_file}")
        else:  # Linux
            os.system(f"xdg-open {output_file}")
    except Exception as e:
        print(f"Error generating speech: {e}")

def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                            (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path, weights_only=True))
    decoder.load_state_dict(torch.load(args.decoder_path, weights_only=True))

    # Prepare an image
    image = load_image(args.image, transform)
    if image is None:
        return  # Exit if image loading failed
    
    image_tensor = image.to(device)
    
    # Generate a caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()  # (1, max_seq_length) -> (max_seq_length)
    
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
            
    # Join the caption and clean it
    sentence = ' '.join(sampled_caption)
    generated_caption_clean = sentence.replace('<start>', '').replace('<end>', '').strip()

    # Print out the image and the generated caption
    print(f"Generated caption for {os.path.basename(args.image)}: {generated_caption_clean}")
    
    translator = Translator()
    generated_caption_hindi = translator.translate(generated_caption_clean, src='en', dest='hi').text
    print(f"Translated Caption: {generated_caption_hindi}")
    
    # Text-to-speech for the generated caption
    text_to_speech_gtts_auto(generated_caption_hindi)
    
    # Optionally, compute BLEU score if a reference caption is provided
    if args.reference_caption:
        reference_caption_clean = args.reference_caption.replace('<start>', '').replace('<end>', '').strip()
        reference_tokens = nltk.tokenize.word_tokenize(reference_caption_clean.lower())
        generated_tokens = nltk.tokenize.word_tokenize(generated_caption_clean.lower())
        
        # Smoothing function
        smoothing_function = SmoothingFunction().method1
        
        # Compute BLEU scores for n-grams 1 to 4
        bleu1 = sentence_bleu([reference_tokens], generated_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing_function)
        bleu2 = sentence_bleu([reference_tokens], generated_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function)
        bleu3 = sentence_bleu([reference_tokens], generated_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_function)
        bleu4 = sentence_bleu([reference_tokens], generated_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)
        
        # Print the BLEU scores
        print(f"BLEU-1: {bleu1:.4f}")
        print(f"BLEU-2: {bleu2:.4f}")
        print(f"BLEU-3: {bleu3:.4f}")
        print(f"BLEU-4: {bleu4:.4f}")

    # Show the image
    image = Image.open(args.image)
    plt.imshow(np.asarray(image))
    plt.axis('off')  # Hide axes
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate captions for images using a pre-trained model.')
    parser.add_argument('--image', type=str, required=True, help='Input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='models/encoder-5-3000.pkl', help='Path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-5-3000.pkl', help='Path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='Path for vocabulary wrapper')
    
    # Model parameters (should be same as parameters in train.py)
    parser.add_argument('--embed_size', type=int, default=256, help='Dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='Dimension of LSTM hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers in LSTM')
    
    # Optional reference caption for BLEU score
    parser.add_argument('--reference_caption', type=str, help='Reference caption for BLEU score calculation')
    
    args = parser.parse_args()
    main(args)
