import os 
import base64
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_qdrant import Qdrant

class EmbeddingsManager:
    def __init__(
            self,
            model_name: str = "BAAI/bge-small-en",
            device: str = "cpu",
            encode_kwargs: dict = {"normalize_embeddings":True},
            qdrant_url: str = "http://localhost:6333",
            collection_name: str = "vector_db"
    ):
        """
        Intializes the EmbeddingsManager with the specified model and Qdrant settings. 

        Args:
            model_name (str): Huggingface model name. 
            device (str): Device used to run the model. 
            encode_kwargs (dict): Additional keywords arguments for encoding.
            qdrant_url (str): The url of the Qdrant instance.
            collection_name (str): The name of the Qdrant db.
        """
        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name

        # Setting up embedding 
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs
        )
    def create_embeddings(self, pdf_path: str):
        """
        Process the PDF, create embeddings, and store them in Qdrant.

        Args:
            path_pdf (str): The file path to the pdf.

        returns:
            str: Success message upon completion
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"The file {pdf_path} does not exits.")
        
        # Intializer Loader class
        loader = UnstructuredPDFLoader(pdf_path)
        # Load documents
        docs = loader.load()
        if not docs:
            raise ValueError(f"No documents were loaded")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=250
        )
        splits = text_splitter.split_documents(docs)
        if not splits:
            raise ValueError(f"No chunks were created from the documents")
        
        # Create and store embeddings in Qdrant
        try:
            qdrant = Qdrant.from_documents(
                splits,
                embedding=self.embeddings,
                url=self.qdrant_url,
                prefer_grpc=False,
                collection_name=self.collection_name,
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant: {e}")
        
        return "Vector DB Successfully Created and Stored in Qdrant!"