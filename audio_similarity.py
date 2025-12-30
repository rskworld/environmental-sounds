"""
Environmental Sound Dataset - Audio Similarity Search Module

Project: Environmental Sound Dataset
Website: https://rskworld.in
Founded by: Molla Samser
Designer & Tester: Rima Khatun
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides audio similarity search and clustering capabilities.
"""

import numpy as np
import librosa
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import pickle
import os


class AudioSimilaritySearch:
    """
    Audio similarity search engine using feature embeddings.
    """
    
    def __init__(self, feature_type: str = 'mfcc', n_features: int = 13):
        """
        Initialize similarity search.
        
        Args:
            feature_type: Type of features to use ('mfcc', 'mel', 'chroma')
            n_features: Number of features
        """
        self.feature_type = feature_type
        self.n_features = n_features
        self.audio_features = []
        self.audio_paths = []
        self.embeddings = None
    
    def extract_features(self, audio: np.ndarray, sr: int = 22050) -> np.ndarray:
        """
        Extract features from audio.
        
        Args:
            audio: Audio waveform
            sr: Sample rate
        
        Returns:
            Feature vector
        """
        if self.feature_type == 'mfcc':
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_features)
            features = np.mean(mfccs, axis=1)
        elif self.feature_type == 'mel':
            mel = librosa.feature.melspectrogram(y=audio, sr=sr)
            features = np.mean(mel, axis=1)
        elif self.feature_type == 'chroma':
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features = np.mean(chroma, axis=1)
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
        
        return features
    
    def add_audio(self, audio: np.ndarray, audio_path: str = None, sr: int = 22050):
        """
        Add audio to the search index.
        
        Args:
            audio: Audio waveform
            audio_path: Path to audio file (optional)
            sr: Sample rate
        """
        features = self.extract_features(audio, sr)
        self.audio_features.append(features)
        self.audio_paths.append(audio_path)
    
    def build_index(self):
        """Build the search index."""
        if len(self.audio_features) == 0:
            raise ValueError("No audio files added to index")
        
        self.embeddings = np.array(self.audio_features)
        print(f"Index built with {len(self.embeddings)} audio files")
    
    def search(
        self,
        query_audio: np.ndarray,
        top_k: int = 5,
        metric: str = 'cosine',
        sr: int = 22050
    ) -> List[Tuple[int, float, str]]:
        """
        Search for similar audio files.
        
        Args:
            query_audio: Query audio waveform
            top_k: Number of results to return
            metric: Similarity metric ('cosine', 'euclidean')
            sr: Sample rate
        
        Returns:
            List of (index, similarity_score, path) tuples
        """
        if self.embeddings is None:
            self.build_index()
        
        query_features = self.extract_features(query_audio, sr)
        query_features = query_features.reshape(1, -1)
        
        if metric == 'cosine':
            similarities = cosine_similarity(query_features, self.embeddings)[0]
        elif metric == 'euclidean':
            distances = euclidean_distances(query_features, self.embeddings)[0]
            similarities = 1 / (1 + distances)  # Convert distance to similarity
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((idx, float(similarities[idx]), self.audio_paths[idx]))
        
        return results
    
    def find_duplicates(self, threshold: float = 0.95) -> List[List[int]]:
        """
        Find duplicate or near-duplicate audio files.
        
        Args:
            threshold: Similarity threshold for duplicates
        
        Returns:
            List of groups of duplicate indices
        """
        if self.embeddings is None:
            self.build_index()
        
        similarities = cosine_similarity(self.embeddings)
        duplicates = []
        processed = set()
        
        for i in range(len(similarities)):
            if i in processed:
                continue
            
            group = [i]
            for j in range(i + 1, len(similarities)):
                if similarities[i][j] >= threshold:
                    group.append(j)
                    processed.add(j)
            
            if len(group) > 1:
                duplicates.append(group)
                processed.add(i)
        
        return duplicates
    
    def save_index(self, filepath: str):
        """Save the search index."""
        data = {
            'embeddings': self.embeddings,
            'audio_paths': self.audio_paths,
            'feature_type': self.feature_type,
            'n_features': self.n_features
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str):
        """Load a saved search index."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings']
        self.audio_paths = data['audio_paths']
        self.feature_type = data['feature_type']
        self.n_features = data['n_features']
        print(f"Index loaded from {filepath}")


class AudioClustering:
    """
    Audio clustering using various algorithms.
    """
    
    def __init__(self, feature_type: str = 'mfcc', n_features: int = 13):
        """
        Initialize audio clustering.
        
        Args:
            feature_type: Type of features to use
            n_features: Number of features
        """
        self.feature_type = feature_type
        self.n_features = n_features
        self.features = []
        self.labels = None
        self.model = None
    
    def extract_features(self, audio_list: List[np.ndarray], sr: int = 22050) -> np.ndarray:
        """
        Extract features from audio list.
        
        Args:
            audio_list: List of audio waveforms
            sr: Sample rate
        
        Returns:
            Feature matrix
        """
        features = []
        for audio in audio_list:
            if self.feature_type == 'mfcc':
                mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_features)
                feat = np.mean(mfccs, axis=1)
            elif self.feature_type == 'mel':
                mel = librosa.feature.melspectrogram(y=audio, sr=sr)
                feat = np.mean(mel, axis=1)
            else:
                raise ValueError(f"Unknown feature type: {self.feature_type}")
            
            features.append(feat)
        
        self.features = np.array(features)
        return self.features
    
    def cluster_kmeans(self, n_clusters: int = 5, random_state: int = 42) -> np.ndarray:
        """
        Perform K-means clustering.
        
        Args:
            n_clusters: Number of clusters
            random_state: Random seed
        
        Returns:
            Cluster labels
        """
        if len(self.features) == 0:
            raise ValueError("No features extracted. Call extract_features first.")
        
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.labels = self.model.fit_predict(self.features)
        return self.labels
    
    def cluster_dbscan(self, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
        """
        Perform DBSCAN clustering.
        
        Args:
            eps: Maximum distance between samples
            min_samples: Minimum samples in a cluster
        
        Returns:
            Cluster labels
        """
        if len(self.features) == 0:
            raise ValueError("No features extracted. Call extract_features first.")
        
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels = self.model.fit_predict(self.features)
        return self.labels
    
    def reduce_dimensions(self, n_components: int = 2) -> np.ndarray:
        """
        Reduce feature dimensions using PCA.
        
        Args:
            n_components: Number of components
        
        Returns:
            Reduced features
        """
        if len(self.features) == 0:
            raise ValueError("No features extracted.")
        
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(self.features)
        return reduced
    
    def get_cluster_statistics(self) -> Dict:
        """
        Get statistics about clusters.
        
        Returns:
            Dictionary with cluster statistics
        """
        if self.labels is None:
            raise ValueError("No clustering performed yet.")
        
        unique_labels = np.unique(self.labels)
        stats = {
            'n_clusters': len(unique_labels),
            'cluster_sizes': {},
            'noise_points': 0
        }
        
        for label in unique_labels:
            count = np.sum(self.labels == label)
            if label == -1:  # DBSCAN noise
                stats['noise_points'] = count
            else:
                stats['cluster_sizes'][f'cluster_{label}'] = count
        
        return stats


if __name__ == '__main__':
    print("Audio Similarity Search Module")
    print("=" * 50)
    print("Features:")
    print("  - AudioSimilaritySearch: Find similar audio files")
    print("  - AudioClustering: Cluster audio by similarity")
    print()
    print("Example usage:")
    print("  search = AudioSimilaritySearch()")
    print("  search.add_audio(audio1, 'path1.wav')")
    print("  results = search.search(query_audio, top_k=5)")

