import math
import torch
from torch import nn


# config = {
#     "patch_size": 4,  # Input image size: 32x32 -> 8x8 patches
#     "hidden_size": 48,
#     "num_hidden_layers": 4,
#     "num_attention_heads": 4,
#     "intermediate_size": 4 * 48, # 4 * hidden_size
#     "hidden_dropout_prob": 0.0,
#     "attention_probs_dropout_prob": 0.0,
#     "initializer_range": 0.02,
#     "image_size": 32,
#     "num_classes": 10, # num_classes of CIFAR10
#     "num_channels": 3,
#     "qkv_bias": True,
#     "use_faster_attention": True,
# }



class NewGELUActivation(nn.Module):
    """
    Gaussian Error Linear Units. Paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.04475 * torch.pow(input, 3.0))))
    

class PatchEmbeddings(nn.Module):
    """
    Convert the images into the patches and then project them into a vector space.
    """

    def __init__(self, config):
        super().__init__()
        self.image_size = config['image_size']
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        # Calculate the number of patches size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # Create a projection layer to convert the image into patches
        # The layer projects each patch intp a vector of hidden_size
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, num_patches, hidden_size)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x
    

class Embeddings(nn.Module):
    """
    Combine the patch embeddings with the class token and position embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)
        # create a learnable [CLS] token
        # Similar to BERT, the [CLS] token is added to the beginning of the input sequence
        # and is used to classify the entire sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        # Create the positional embedding for the [CLS] token and the patch embeddings
        # Add 1 to the sequence length for the [CLS] token
        self.position_embeddings = nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"]))
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        # Expand the [CLS] token to the batch_size
        # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # Concatenate the [CLS] token to the beginning of the input sequencs in each batch
        # This results in a sequence length of (num_patches + 1) in each batch
        x = torch.cat((cls_tokens, x), dim=1) # (batch_size, num_patches + 1, hidden_size)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x
    

class AttentionHead(nn.Module):
    """
    A single attention head.
    This will be used in the MultiHeadAttention module.
    """

    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        # Create QKV projection layers
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Project the input into query, key and value
        # The same input is used to generate the query, key and value. hence the name (self-attention)
        # (batch_size, sequence_length, hiiden_size) -> (batch_size, sequence_length, attention_head_size)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # Calculate the attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        # Calculate the attention outputs
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)
    

class MultiHeadAttention(nn.Module):
    """
    This module is used in the TransformerEncoder Module
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Whether or not to use the bias in the QKC projection layers
        self.qkv_bias = config["qkv_bias"]
        # Create a list of attention heads
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.qkv_bias
            )
            self.heads.append(head)
        # Create a linear lauer to poject the attention output back to the hidden size
        # In most cases all_head_size and the hidden_size are same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # Calculate the attention output for each attention head
        attention_outputs = [head(x) for head in self.heads]
        # Concatenate the attention output from each attention heads
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)
        # Project the concatenated attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output and attention probabilites (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim=1)
            return (attention_output, attention_probs)
        

class FasterMultiHeadAttention(nn.Module):
    """
    All the heads are processed simultaneously with merged query, key and value projections.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.attention_head_size * self.num_attention_heads
        self.qkv_bias = config["qkv_bias"]
        self.qkv_projection = nn.Linear(self.hidden_size, self.all_head_size * 3, bias=self.qkv_bias)
        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, all_head_size * 3)
        qkv = self.qkv_projection(x)
        # (batch_size, sequence_length, all_head_size * 3) -> (batch_size, sequence_length, all_head_size)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        batch_size, sequence_length, _ = query.size()
        query = query.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        attention_output = torch.matmul(attention_probs, value)
        # (b, h, t, c_h) -> (b, t, h * c_h)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.all_head_size)
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        if not output_attentions:
            return ( attention_output, None)
        else:
            return (attention_output, attention_probs)
        

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation = NewGELUActivation()
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        return self.dropout(self.dense_2(self.activation(self.dense_1(x))))
    

class Block(nn.Module):
    """
    A single transformer block
    """

    def __init__(self, config):
        super().__init__()
        self.use_faster_attention = config.get("use_faster_attention", True)
        if self.use_faster_attention:
            self.attention = FasterMultiHeadAttention(config)
        else:
            self.attention = MultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x, output_attentions=False):
        # Self attention
        attention_output, attention_probs = self.attention(self.layernorm_1(x), output_attentions=output_attentions)
        # Residual connection
        x = x + attention_output
        # Feed forward network
        mlp_output = self.mlp(self.layernorm_2(x))
        # Skip connection
        x = x + mlp_output
        # Return the transformer block's output and attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)
        

class Encoder(nn.Module):
    """
    The transformer encoder module.
    """

    def __init__(self, config):
        super().__init__()
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = Block(config)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        # Calculate the transformer block's output for each block
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
        # Return the encoder's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)
        

class ViTForClassification(nn.Module):
    """
    The ViT for classification
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        # Create the embedding module
        self.embedding = Embeddings(config)
        # Create the transformer encoder module
        self.encoder = Encoder(config)
        # Create a linear layer to project the encoder's output to the number of classes
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        # Initialize the weights
        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False):
        # Calculate the embedding output
        embedding_output = self.embedding(x)
        # Calculate the encoder's output
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
        # Calculate the logits, take the [CLS] token's output as feature for the classification
        logits = self.classifier(encoder_output[:, 0, :])
        # return the logits and the attention probabilites (optional)
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean = 0.0,
                std = self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean = 0.0,
                std = self.config["initializer_range"],
            ).to(module.cls_token.dtype)


# def parse_args():
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--exp-name", type=str, required=True)
#     parser.add_argument("--batch-size", type=int, default=256)
#     parser.add_argument("--epochs", type=int, default=100)
#     parser.add_argument("--lr", type=float, default=1e-2)
#     parser.add_argument("--device", type=str)
#     parser.add_argument("--save-model-every", type=int, default=0)

#     args = parser.parse_args()
#     if args.device is None:
#         args.device = "cuda" if torch.cuda.is_available() else "cpu"
#     return args


# def main():
#     args = parse_args()
#     # Training parameters
#     batch_size = args.batch_size
#     epochs = args.epochs
#     lr = args.lr
#     device = args.device
#     save_model_every_n_epochs = args.save_model_every
#     x = torch.rand(batch_size, config["num_channels"], config["image_size"], config["image_size"]).to(device)
#     model = ViTForClassification(config=config).to(device)

#     # Output the shape of the logits
#     output, attentions = model(x, output_attentions=True)

#     # If you want to inspect the attention maps, you can also print their shapes
#     if attentions is not None:
#         print("Number of attention heads:", len(attentions))
#         print("Attention map shape for each head:", attentions[1].shape)
#     print(output.shape)


# if __name__ == "__main__":
#     main()