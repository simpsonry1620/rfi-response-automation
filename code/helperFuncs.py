import os
from sentence_transformers import SentenceTransformer
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource, HoverTool, WheelZoomTool
from bokeh.palettes import Category10
from sklearn.manifold import TSNE
import umap.umap_ as umap
from torch.nn.functional import normalize
from langchain.embeddings.base import Embeddings
from ipywidgets import FileUpload
from IPython.display import display
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from bokeh.io import output_notebook
import numpy as np

#Used to load in .txt, .pdf, .doc, and .docx files, only handles text, not tables or images.
def load_documents(directory, docs=[]):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith('.txt'):
            loader = TextLoader(filepath)
            docs.extend(loader.load())
        elif filename.endswith('.pdf'):
            loader = PyPDFLoader(filepath)
            docs.extend(loader.load())
        elif filename.endswith(('.doc', '.docx')):
            loader = Docx2txtLoader(filepath)
            docs.extend(loader.load())
    return docs

def addTags(docs):
    for doc in docs:
        metadata = doc.metadata
        doc_label = metadata.get("source")
        if 'partner-docs' in doc_label:
            doc.metadata['tag'] = 'Partner'
        elif 'rfi-docs' in doc_label:
            doc.metadata['tag'] = 'RFI'
        else: 
            doc.metadata['tag'] = 'NVIDIA'

def save_uploaded_files(change, target_dir):
    
    # Create the directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Iterate over each uploaded file
    for i, file in enumerate(file_upload.value):
        content = file['content']
        filename = file['name']
        filepath = os.path.join(target_dir, filename)
        
        # Save the file
        with open(filepath, 'wb') as f:
            f.write(content)
        
        print(f"Saved {filename} to {filepath}")
        
    # Clear the file upload widget after saving files
    file_upload.value = []

def create_callback(target_dir):
    def callback(change):
        save_uploaded_files(change, target_dir)
    return callback

def rerank_results(query: str, documents: list, top_k: int = 10):
    # Initialize Longformer model and tokenizer
    model_name = "allenai/longformer-base-4096"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
    
    # Move model to appropriate device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    # Create pairs and get scores
    pairs = []
    chunk_size = int(os.environ['CHUNK_SIZE'])
    for doc in documents:
        # Format input as expected by Longformer
        inputs = tokenizer(
            query,
            doc.page_content,
            padding=True,
            truncation=True,
            max_length=chunk_size,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Get probability for positive class
            scores = F.softmax(outputs.logits, dim=1)[:, 1]
            pairs.append((doc, scores.item()))
    
    # Sort by scores in descending order
    pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Return top_k documents
    return [doc for doc, _ in pairs[:top_k]]


# RAG function with vector store option
def rag_query(query, model, vectorstore=None, rerank=False, show_docs=False, top_k=50, tag='tag', tag_values=['NVIDIA', 'Partner']):
    if vectorstore:
        # Retrieve relevant documents
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": top_k})
        docs = retriever.invoke(query, filter={tag: {"$in": tag_values}})
        
        # potential to include Rerank
        if rerank:
            reranked_docs = rerank_results(query, docs, top_k=top_k//2)
        else:
            reranked_docs = docs[:top_k]
        context = "\n".join([doc.page_content for doc in reranked_docs])
    else:
        context = ""
    
    # Generate response using the LLM
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = model.invoke(prompt)

    if show_docs:
        return response, reranked_docs
    
    return response


class M2BERTEmbeddings(Embeddings):
    def __init__(
        self,
        model_name: str = "togethercomputer/m2-bert-80M-2k-retrieval",
        max_seq_length: int = 2048,
        device: str = ("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.device = device
        print(f"Using device: {self.device}")
        
        # Initialize model with output_hidden_states=True
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            output_hidden_states=True  # Enable hidden states output
        ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            model_max_length=max_seq_length
        )
        
        self.max_seq_length = max_seq_length
        self.model.eval()
    
    def _get_embedding(self, text):
        # Tokenize text
        inputs = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Get the last hidden state
            embeddings = outputs['sentence_embedding']
        
        return embeddings[0].tolist()
    
    def embed_documents(self, texts):
        """Generate embeddings for a list of documents."""
        return [self._get_embedding(text) for text in texts]
    
    def embed_query(self, text):
        """Generate embedding for a query."""
        return self._get_embedding(text)

def bokeh_plot(data, annotations, classes, colors, title):
    data = {
    'x': data[:, 0],
    'y': data[:, 1],
    'filename': annotations,
    'label': classes,
    'color': colors  # Add color information to the data source
    }
    source = ColumnDataSource(data)
    
    # Create a Bokeh figure
    p = figure(
        title=title,
        tools="pan,wheel_zoom,reset",
        width=800,
        height=600,
        tooltips=[("Filename", "@filename")]
    )
    
    # Add scatter points with color and legend grouping by 'label'
    p.scatter('x', 'y', size=8, source=source, fill_alpha=0.6, color='color', legend_field='label')
    
    # Customize the legend
    p.legend.title = "Labels"
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"  # Allow interactive toggling of labels

    wheel_zoom_tool = p.select_one(WheelZoomTool)
    p.toolbar.active_scroll = wheel_zoom_tool
    
    hover_tool = p.select_one(HoverTool)
    p.toolbar.active_inspect = [hover_tool]
    
    # Show the plot
    show(p)

def extract_embeddings_and_metadata(vectorstore):
    """Extract embeddings, labels and filenames from vectorstore"""
    index = vectorstore.index
    docstore = vectorstore.docstore
    num_docs = len(docstore._dict)
    
    embeddings = []
    labels = []
    filenames = []
    
    for i in range(num_docs):
        emb = index.reconstruct(i)
        embeddings.append(emb)
        doc_id = vectorstore.index_to_docstore_id[i]
        metadata = docstore.search(str(doc_id)).metadata
        doc_label = metadata.get("source", f"doc_{i}")
        
        if 'partner-docs' in doc_label:
            filenames.append(doc_label.split('/')[-1].split('.')[0])
            label = 'Partner'
        elif 'rfi-docs' in doc_label:
            filenames.append(doc_label.split('/')[-1].split('.')[0])
            label = 'RFI'
        else:
            label = 'NVIDIA'
            filenames.append(doc_label.split('_')[-1].split('.')[0])
        labels.append(label)
    
    return np.array(embeddings), labels, filenames

def get_color_mapping(labels):
    """Create color mapping for unique labels"""
    unique_labels = list(set(labels))
    color_palette = Category10[10]
    label_to_color = {label: color_palette[i] for i, label in enumerate(unique_labels)}
    colors = [label_to_color[label] for label in labels]
    return colors

def visualize_embeddings(vectorstore):
    """Main visualization function"""
    output_notebook()
    
    # Extract data
    embeddings, labels, filenames = extract_embeddings_and_metadata(vectorstore)
    colors = get_color_mapping(labels)
    
    # t-SNE visualization
    tsne = TSNE(
        n_components=2,
        perplexity=min(40, len(embeddings) - 1),
        n_iter=2000, 
        random_state=42,
        learning_rate='auto',  # Auto learning rate
        init='pca'  # PCA initialization
    )
    tsne_embeddings = tsne.fit_transform(embeddings)
    tsne_title = "t-SNE Visualization of Embeddings"
    bokeh_plot(tsne_embeddings, filenames, labels, colors, tsne_title)
    
    # UMAP visualization
    umap_reducer = umap.UMAP(
        n_neighbors=min(30, len(embeddings) - 1),
        n_components=2,
        metric='cosine',
        min_dist=0.1,
        random_state=42
    )
    umap_embeddings = umap_reducer.fit_transform(embeddings)
    umap_title = "UMAP Visualization of Embeddings"
    bokeh_plot(umap_embeddings, filenames, labels, colors, umap_title)


