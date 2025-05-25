import pandas as pd
import numpy as np

# Generate synthetic data
np.random.seed(42)

# Sample texts
texts = [
    "This product is amazing! I love it.",
    "The service was terrible. I'm never coming back.",
    "Neutral review without strong opinions.",
    "I had high hopes, but it fell short.",
    "Great experience overall!",
    "Not worth the money.",
    "It works well, but could be better.",
    "Average performance, nothing special.",
    "Completely satisfied with my purchase.",
    "Disappointed with the quality.",
    "Excellent customer service.",
    "Just what I needed, thank you!",
    "Could have been better.",
    "Really bad experience, would not recommend.",
    "Met my expectations perfectly.",
    "Could not be happier with this product.",
    "Poor customer support.",
    "Decent product for the price.",
    "Very impressed with the design.",
    "Waste of money.",
    "Impartial review.",
    "Satisfied with the outcome.",
    "Highly recommend!",
    "Not up to my standards.",
    "Very pleased with the results.",
    "Too expensive for what it offers.",
    "Mediocre performance.",
    "Absolutely delighted with this purchase.",
    "Frustrating experience dealing with them.",
    "Fairly good product.",
    "Nothing exceptional about it.",
    "Thrilled with the quality.",
    "Awful service, avoid at all costs.",
    "Reasonably happy with it.",
    "Didn't meet my expectations.",
    "Top-notch service!",
    "Could improve in many aspects.",
    "Pleasantly surprised by its performance.",
    "Horrible product, regret buying it.",
    "Okay but not great.",
    "Fantastic product!",
    "Not reliable.",
    "Could be more user-friendly.",
    "Outstanding customer service.",
    "Not recommended.",
    "Impressive performance.",
    "Really disappointed with the outcome.",
    "Well worth the price.",
    "Underwhelmed by its performance.",
    "Great value for money.",
    "Unsatisfactory experience.",
    "Impartial opinion.",
]

# Sample sentiments
sentiments = [
    "positive", "negative", "neutral",
    "negative", "positive", "negative",
    "neutral", "neutral", "positive",
    "negative", "positive", "positive",
    "negative", "negative", "positive",
    "positive", "negative", "neutral",
    "positive", "negative", "neutral",
    "positive", "negative", "positive",
    "positive", "negative", "neutral",
    "negative", "positive", "neutral",
    "neutral", "positive", "negative",
    "positive", "negative", "positive",
    "negative", "neutral", "positive",
    "negative", "positive", "negative",
    "positive", "negative", "positive",
    "negative", "positive", "negative",
    "positive", "negative", "neutral",
]

texts = texts[:-1]

# Create a DataFrame
data=pd.DataFrame({
    'text': texts,
    'sentiment': sentiments
})

# Save to CSV
data.to_csv('sample_data.csv', index=False)

print(f'Saved {len(data)} samples to sample_data.csv')

# Import necessary libraries
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Example data (replace this with your dataset loading code)
data = pd.read_csv('sample_data.csv')
data = data.sample(frac=1).reset_index(drop=True)  # Shuffle data

# Map sentiment labels to integers
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
data['label'] = data['sentiment'].map(label_map)

# Tokenize text data
input_ids = []
attention_masks = []

for sent in data['text']:
    encoded_dict = tokenizer.encode_plus(
                        sent,
                        add_special_tokens = True,
                        max_length = 64,
                        padding = 'max_length',
                        truncation = True,
                        return_attention_mask = True,
                        return_tensors = 'pt',
                   )

    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(data['label'].values)

# Split data into training and validation sets
train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = train_test_split(input_ids, attention_masks, labels,
                                                                                                random_state=2022, test_size=0.1)

# Convert to PyTorch DataLoader
batch_size = 32

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = 3,  # 3 sentiment classes (negative, neutral, positive)
    output_attentions = False,
    output_hidden_states = False,
)

# Define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)
epochs = 3
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

# Function to evaluate model performance
def evaluate(model, val_dataloader):
    model.eval()
    val_loss = 0
    predictions, true_vals = [], []

    for batch in val_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs, masks, labels = batch

        with torch.no_grad():
            outputs = model(inputs, token_type_ids=None, attention_mask=masks)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()

        predictions.append(logits)
        true_vals.append(label_ids)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return predictions, true_vals

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(epochs):
    model.train()

    for batch in tqdm(train_dataloader, desc='Training'):
        batch = tuple(t.to(device) for t in batch)
        inputs, masks, labels = batch

        model.zero_grad()
        outputs = model(inputs, token_type_ids=None, attention_mask=masks, labels=labels)

        loss = outputs[0]
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

# Evaluation
predictions, true_vals = evaluate(model, val_dataloader)
preds_flat = np.argmax(predictions, axis=1).flatten()
true_vals_flat = true_vals.flatten()

# Calculate accuracy
accuracy = accuracy_score(true_vals_flat, preds_flat)
print(f'Accuracy: {accuracy}')

# Classification report
print(classification_report(true_vals_flat, preds_flat, target_names=label_map.keys()))


# In[3]:





# In[2]:


import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

# ... (rest of your existing code)
test_sentence = "Completely satisfied with my purchase"

# Tokenize input
encoded_dict = tokenizer.encode_plus(
    test_sentence,
    add_special_tokens=True,
    max_length=64,
    padding='max_length',
    truncation=True,
    return_attention_mask=True,
    return_tensors='pt'  # Return PyTorch tensors
)

# Prepare input for the model
input_ids = encoded_dict['input_ids'].to(device)
attention_mask = encoded_dict['attention_mask'].to(device)

# Get model prediction
model.eval()  # Set model to evaluation mode
with torch.no_grad():  # Disable gradient calculation
    outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask)

# Extract logits and apply softmax
logits = outputs[0]
probabilities = torch.nn.functional.softmax(logits, dim=1)

# Get predicted label
predicted_label_idx = torch.argmax(probabilities, dim=1).item()

# Map label index to label name
label_map_inverse = {v: k for k, v in label_map.items()}
predicted_label = label_map_inverse[predicted_label_idx]

print(f"Predicted Sentiment: {predicted_label}")





