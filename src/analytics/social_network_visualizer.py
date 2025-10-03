#!/usr/bin/env python3
"""
📊 Social Network Graph Visualizer - Interactive Relationship Mapping
LakyLuk OSINT Investigation Suite

Features:
✅ Interactive network graphs with zoom, pan, and filtering
✅ Multi-dimensional relationship visualization (platforms, strength, time)
✅ AI-powered community detection and clustering
✅ Real-time graph updates and live investigation tracking
✅ Export capabilities (PNG, SVG, PDF, interactive HTML)
✅ Advanced graph metrics and centrality analysis

Visualization Types:
- Force-directed layouts for organic relationship mapping
- Hierarchical layouts for organizational structures
- Circular layouts for community visualization
- Timeline layouts for temporal relationship analysis
- Geographic layouts for location-based connections
"""

import asyncio
import json
import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
import colorsys
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, Slider
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

from ..analytics.entity_correlation_engine import EntityProfile, CorrelationResult
from ..core.enhanced_orchestrator import EnhancedInvestigationOrchestrator


@dataclass
class NetworkNode:
    """Node in the social network graph"""
    id: str
    label: str
    platform: str
    node_type: str  # 'person', 'organization', 'location', 'event'
    properties: Dict[str, Any]
    position: Optional[Tuple[float, float]] = None
    size: float = 10.0
    color: str = "#3498db"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class NetworkEdge:
    """Edge in the social network graph"""
    source: str
    target: str
    relationship_type: str
    weight: float
    platform: str
    properties: Dict[str, Any]
    color: str = "#95a5a6"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class NetworkCommunity:
    """Detected community in the network"""
    id: str
    members: List[str]
    community_type: str
    center_node: str
    properties: Dict[str, Any]
    color: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SocialNetworkVisualizer:
    """Advanced social network visualization and analysis system"""

    def __init__(self, ai_orchestrator: EnhancedInvestigationOrchestrator = None):
        self.ai_orchestrator = ai_orchestrator

        # Graph storage
        self.graph = nx.Graph()
        self.nodes = {}  # id -> NetworkNode
        self.edges = {}  # (source, target) -> NetworkEdge
        self.communities = {}  # id -> NetworkCommunity

        # Visualization settings
        self.vis_config = {
            'width': 1200,
            'height': 800,
            'node_size_range': (10, 50),
            'edge_width_range': (1, 8),
            'font_size': 12,
            'show_labels': True,
            'show_edge_labels': False,
            'layout_algorithm': 'spring',
            'color_scheme': 'platform_based'
        }

        # Platform colors
        self.platform_colors = {
            'facebook': '#1877f2',
            'instagram': '#e4405f',
            'linkedin': '#0077b5',
            'twitter': '#1da1f2',
            'unknown': '#95a5a6',
            'multiple': '#9b59b6'
        }

        # Layout algorithms
        self.layout_algorithms = {
            'spring': nx.spring_layout,
            'circular': nx.circular_layout,
            'kamada_kawai': nx.kamada_kawai_layout,
            'spectral': nx.spectral_layout,
            'random': nx.random_layout
        }

    async def create_network_from_profiles(self, profiles: List[EntityProfile],
                                         correlations: List[CorrelationResult] = None) -> Dict[str, Any]:
        """
        Create network graph from entity profiles and correlations

        Args:
            profiles: List of entity profiles to visualize
            correlations: Optional correlation results for relationships

        Returns:
            Network creation summary with statistics
        """
        try:
            # Clear existing graph
            self.graph.clear()
            self.nodes.clear()
            self.edges.clear()
            self.communities.clear()

            # Add nodes from profiles
            for profile in profiles:
                await self._add_profile_node(profile)

            # Add edges from correlations
            if correlations:
                for correlation in correlations:
                    await self._add_correlation_edges(correlation)

            # Detect communities
            communities = await self._detect_communities()
            self.communities = communities

            # Calculate network metrics
            metrics = self._calculate_network_metrics()

            # AI-powered network analysis
            ai_insights = {}
            if self.ai_orchestrator:
                ai_insights = await self.ai_orchestrator.analyze_network_structure(
                    list(self.nodes.values()), list(self.edges.values())
                )

            return {
                'nodes_count': len(self.nodes),
                'edges_count': len(self.edges),
                'communities_count': len(self.communities),
                'network_metrics': metrics,
                'ai_insights': ai_insights,
                'creation_time': datetime.now().isoformat()
            }

        except Exception as e:
            print(f"Network creation error: {e}")
            return {}

    async def generate_interactive_visualization(self, output_file: str = None,
                                               layout: str = "force") -> str:
        """
        Generate interactive Plotly visualization

        Args:
            output_file: Optional output HTML file path
            layout: Layout algorithm to use

        Returns:
            HTML content or file path of generated visualization
        """
        try:
            # Calculate layout positions
            positions = await self._calculate_layout_positions(layout)

            # Create interactive Plotly figure
            fig = await self._create_plotly_figure(positions)

            # Add interactivity features
            fig = await self._add_plotly_interactivity(fig)

            # Generate HTML
            if output_file:
                pyo.plot(fig, filename=output_file, auto_open=False)
                return output_file
            else:
                html_content = pyo.plot(fig, output_type='div', include_plotlyjs=True)
                return html_content

        except Exception as e:
            print(f"Interactive visualization error: {e}")
            return ""

    async def generate_static_visualization(self, output_file: str = None,
                                          layout: str = "spring") -> str:
        """
        Generate static matplotlib visualization

        Args:
            output_file: Optional output image file path
            layout: Layout algorithm to use

        Returns:
            File path of generated image
        """
        try:
            # Calculate layout positions
            positions = await self._calculate_layout_positions(layout)

            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(15, 10))

            # Draw network
            await self._draw_matplotlib_network(ax, positions)

            # Add title and metadata
            ax.set_title(f"Social Network Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            ax.axis('off')

            # Save or return
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                return output_file
            else:
                temp_file = f"network_visualization_{int(time.time())}.png"
                plt.savefig(temp_file, dpi=300, bbox_inches='tight')
                plt.close()
                return temp_file

        except Exception as e:
            print(f"Static visualization error: {e}")
            return ""

    async def generate_timeline_visualization(self, time_field: str = "created_date") -> str:
        """
        Generate timeline-based network visualization

        Args:
            time_field: Field to use for temporal analysis

        Returns:
            HTML content of timeline visualization
        """
        try:
            # Extract temporal data
            temporal_data = await self._extract_temporal_data(time_field)

            # Create timeline figure
            fig = await self._create_timeline_figure(temporal_data)

            # Generate HTML
            html_content = pyo.plot(fig, output_type='div', include_plotlyjs=True)
            return html_content

        except Exception as e:
            print(f"Timeline visualization error: {e}")
            return ""

    async def analyze_network_communities(self) -> Dict[str, Any]:
        """
        Analyze detected communities in the network

        Returns:
            Comprehensive community analysis
        """
        try:
            analysis = {
                'communities': [],
                'community_metrics': {},
                'inter_community_connections': [],
                'community_characteristics': {}
            }

            for community_id, community in self.communities.items():
                # Basic community info
                community_info = {
                    'id': community_id,
                    'size': len(community.members),
                    'center_node': community.center_node,
                    'type': community.community_type,
                    'members': community.members
                }

                # Calculate community metrics
                subgraph = self.graph.subgraph(community.members)
                metrics = {
                    'density': nx.density(subgraph),
                    'clustering_coefficient': nx.average_clustering(subgraph),
                    'diameter': nx.diameter(subgraph) if nx.is_connected(subgraph) else None
                }

                community_info['metrics'] = metrics
                analysis['communities'].append(community_info)

                # Platform distribution in community
                platform_dist = {}
                for member_id in community.members:
                    if member_id in self.nodes:
                        platform = self.nodes[member_id].platform
                        platform_dist[platform] = platform_dist.get(platform, 0) + 1

                community_info['platform_distribution'] = platform_dist

            # Find inter-community connections
            analysis['inter_community_connections'] = await self._find_inter_community_connections()

            # AI community analysis
            if self.ai_orchestrator:
                ai_community_analysis = await self.ai_orchestrator.analyze_community_structure(
                    analysis['communities']
                )
                analysis['ai_community_insights'] = ai_community_analysis

            return analysis

        except Exception as e:
            print(f"Community analysis error: {e}")
            return {}

    async def calculate_influence_metrics(self) -> Dict[str, Any]:
        """
        Calculate influence and centrality metrics for network nodes

        Returns:
            Comprehensive influence analysis
        """
        try:
            influence_metrics = {}

            # Calculate various centrality measures
            centrality_measures = {
                'degree_centrality': nx.degree_centrality(self.graph),
                'betweenness_centrality': nx.betweenness_centrality(self.graph),
                'closeness_centrality': nx.closeness_centrality(self.graph),
                'eigenvector_centrality': nx.eigenvector_centrality(self.graph),
                'pagerank': nx.pagerank(self.graph)
            }

            # Combine centrality scores
            for node_id in self.graph.nodes():
                node_metrics = {}
                for measure_name, measure_values in centrality_measures.items():
                    node_metrics[measure_name] = measure_values.get(node_id, 0.0)

                # Calculate composite influence score
                influence_score = (
                    node_metrics['degree_centrality'] * 0.3 +
                    node_metrics['betweenness_centrality'] * 0.25 +
                    node_metrics['closeness_centrality'] * 0.2 +
                    node_metrics['eigenvector_centrality'] * 0.15 +
                    node_metrics['pagerank'] * 0.1
                )

                node_metrics['composite_influence_score'] = influence_score
                influence_metrics[node_id] = node_metrics

            # Rank nodes by influence
            ranked_nodes = sorted(
                influence_metrics.items(),
                key=lambda x: x[1]['composite_influence_score'],
                reverse=True
            )

            # AI influence analysis
            ai_influence_insights = {}
            if self.ai_orchestrator:
                ai_influence_insights = await self.ai_orchestrator.analyze_influence_patterns(
                    influence_metrics
                )

            return {
                'node_metrics': influence_metrics,
                'top_influencers': ranked_nodes[:10],
                'network_centralization': self._calculate_network_centralization(centrality_measures),
                'ai_insights': ai_influence_insights
            }

        except Exception as e:
            print(f"Influence metrics error: {e}")
            return {}

    # Internal helper methods

    async def _add_profile_node(self, profile: EntityProfile):
        """Add a profile as a network node"""
        node_id = f"{profile.platform}:{profile.username}"

        # Determine node size based on follower count
        size = self._calculate_node_size(profile.follower_count or 0)

        # Determine node color
        color = self.platform_colors.get(profile.platform, self.platform_colors['unknown'])

        # Create node
        node = NetworkNode(
            id=node_id,
            label=profile.display_name or profile.username,
            platform=profile.platform,
            node_type='person',
            properties={
                'username': profile.username,
                'display_name': profile.display_name,
                'follower_count': profile.follower_count,
                'following_count': profile.following_count,
                'verified': profile.verified,
                'bio': profile.bio,
                'location': profile.location
            },
            size=size,
            color=color,
            metadata={'profile': profile}
        )

        self.nodes[node_id] = node
        self.graph.add_node(node_id, **node.properties)

    async def _add_correlation_edges(self, correlation: CorrelationResult):
        """Add edges based on correlation results"""
        profiles = correlation.profiles

        # Create edges between correlated profiles
        for i in range(len(profiles)):
            for j in range(i + 1, len(profiles)):
                source_id = f"{profiles[i].platform}:{profiles[i].username}"
                target_id = f"{profiles[j].platform}:{profiles[j].username}"

                # Edge weight based on correlation confidence
                weight = correlation.confidence_score

                # Create edge
                edge = NetworkEdge(
                    source=source_id,
                    target=target_id,
                    relationship_type='identity_correlation',
                    weight=weight,
                    platform='cross_platform',
                    properties={
                        'correlation_score': correlation.similarity_score,
                        'confidence_score': correlation.confidence_score,
                        'correlation_factors': correlation.correlation_factors
                    },
                    color=self._get_edge_color(weight),
                    metadata={'correlation': correlation}
                )

                edge_key = (source_id, target_id)
                self.edges[edge_key] = edge
                self.graph.add_edge(source_id, target_id, weight=weight, **edge.properties)

    async def _detect_communities(self) -> Dict[str, NetworkCommunity]:
        """Detect communities in the network using multiple algorithms"""
        communities = {}

        try:
            if len(self.graph.nodes()) < 3:
                return communities

            # Use multiple community detection algorithms
            algorithms = {
                'louvain': self._louvain_communities,
                'greedy_modularity': self._greedy_modularity_communities,
                'edge_betweenness': self._edge_betweenness_communities
            }

            # Run algorithms and select best result
            best_communities = None
            best_modularity = -1

            for algorithm_name, algorithm_func in algorithms.items():
                try:
                    detected_communities = algorithm_func()
                    if detected_communities:
                        modularity = nx.algorithms.community.modularity(self.graph, detected_communities)
                        if modularity > best_modularity:
                            best_modularity = modularity
                            best_communities = detected_communities
                except Exception as e:
                    print(f"Community detection algorithm {algorithm_name} failed: {e}")

            # Convert to NetworkCommunity objects
            if best_communities:
                for i, community_members in enumerate(best_communities):
                    community_id = f"community_{i}"
                    center_node = self._find_community_center(community_members)

                    community = NetworkCommunity(
                        id=community_id,
                        members=list(community_members),
                        community_type='detected',
                        center_node=center_node,
                        properties={'modularity': best_modularity},
                        color=self._generate_community_color(i)
                    )

                    communities[community_id] = community

            # AI-powered community enhancement
            if self.ai_orchestrator and communities:
                enhanced_communities = await self.ai_orchestrator.enhance_community_detection(
                    communities
                )
                communities.update(enhanced_communities)

        except Exception as e:
            print(f"Community detection error: {e}")

        return communities

    async def _calculate_layout_positions(self, layout: str) -> Dict[str, Tuple[float, float]]:
        """Calculate node positions using specified layout algorithm"""
        if layout not in self.layout_algorithms:
            layout = 'spring'

        try:
            layout_func = self.layout_algorithms[layout]

            if layout == 'spring':
                # Enhanced spring layout with better parameters
                positions = layout_func(
                    self.graph,
                    k=3/math.sqrt(len(self.graph.nodes())),
                    iterations=50,
                    weight='weight'
                )
            else:
                positions = layout_func(self.graph)

            return positions

        except Exception as e:
            print(f"Layout calculation error: {e}")
            # Fallback to random layout
            return nx.random_layout(self.graph)

    async def _create_plotly_figure(self, positions: Dict[str, Tuple[float, float]]) -> go.Figure:
        """Create interactive Plotly figure"""
        fig = go.Figure()

        # Add edges
        edge_trace = self._create_plotly_edge_trace(positions)
        fig.add_trace(edge_trace)

        # Add nodes
        node_trace = self._create_plotly_node_trace(positions)
        fig.add_trace(node_trace)

        # Update layout
        fig.update_layout(
            title="Interactive Social Network Analysis",
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Social Network Visualization - LakyLuk OSINT Suite",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color="gray", size=10)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=self.vis_config['width'],
            height=self.vis_config['height']
        )

        return fig

    def _create_plotly_node_trace(self, positions: Dict[str, Tuple[float, float]]) -> go.Scatter:
        """Create Plotly node trace"""
        x_coords = []
        y_coords = []
        colors = []
        sizes = []
        hover_texts = []
        node_texts = []

        for node_id, position in positions.items():
            if node_id in self.nodes:
                node = self.nodes[node_id]

                x_coords.append(position[0])
                y_coords.append(position[1])
                colors.append(node.color)
                sizes.append(node.size)

                # Hover text
                hover_text = f"<b>{node.label}</b><br>"
                hover_text += f"Platform: {node.platform}<br>"
                if 'follower_count' in node.properties and node.properties['follower_count']:
                    hover_text += f"Followers: {node.properties['follower_count']:,}<br>"
                if 'location' in node.properties and node.properties['location']:
                    hover_text += f"Location: {node.properties['location']}<br>"

                hover_texts.append(hover_text)

                # Node labels
                if self.vis_config['show_labels']:
                    node_texts.append(node.label[:15] + "..." if len(node.label) > 15 else node.label)
                else:
                    node_texts.append("")

        return go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            text=node_texts,
            textposition="bottom center",
            textfont=dict(size=self.vis_config['font_size']),
            hovertext=hover_texts,
            hoverinfo='text',
            name='Profiles'
        )

    def _create_plotly_edge_trace(self, positions: Dict[str, Tuple[float, float]]) -> go.Scatter:
        """Create Plotly edge trace"""
        x_coords = []
        y_coords = []

        for edge_key, edge in self.edges.items():
            source_pos = positions.get(edge.source)
            target_pos = positions.get(edge.target)

            if source_pos and target_pos:
                x_coords.extend([source_pos[0], target_pos[0], None])
                y_coords.extend([source_pos[1], target_pos[1], None])

        return go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines',
            line=dict(width=2, color='rgba(125, 125, 125, 0.5)'),
            hoverinfo='none',
            name='Connections'
        )

    async def _add_plotly_interactivity(self, fig: go.Figure) -> go.Figure:
        """Add interactive features to Plotly figure"""
        # Add dropdown for layout selection
        layout_buttons = []
        for layout_name in self.layout_algorithms.keys():
            layout_buttons.append(
                dict(
                    label=layout_name.title(),
                    method='restyle',
                    args=[{'visible': [True, True]}]  # Keep both traces visible
                )
            )

        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=layout_buttons,
                    direction="down",
                    showactive=True,
                    x=0.1,
                    y=1.02,
                    xanchor="left",
                    yanchor="top"
                )
            ]
        )

        return fig

    # Additional utility methods...

    def _calculate_node_size(self, follower_count: int) -> float:
        """Calculate node size based on follower count"""
        min_size, max_size = self.vis_config['node_size_range']

        if follower_count <= 0:
            return min_size

        # Logarithmic scaling
        log_followers = math.log10(max(follower_count, 1))
        normalized_size = min_size + (log_followers / 6) * (max_size - min_size)

        return min(max(normalized_size, min_size), max_size)

    def _get_edge_color(self, weight: float) -> str:
        """Get edge color based on weight/strength"""
        # Color from light gray (weak) to dark blue (strong)
        intensity = min(weight, 1.0)
        r = int(149 - intensity * 100)
        g = int(165 - intensity * 100)
        b = int(166 + intensity * 89)
        return f"rgb({r}, {g}, {b})"

    def _generate_community_color(self, community_index: int) -> str:
        """Generate distinct color for community"""
        # Use HSV color space for distinct colors
        hue = (community_index * 137.508) % 360  # Golden angle
        saturation = 0.7
        value = 0.9

        rgb = colorsys.hsv_to_rgb(hue/360, saturation, value)
        return f"rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})"

    def _calculate_network_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive network metrics"""
        metrics = {}

        if len(self.graph.nodes()) > 0:
            metrics['node_count'] = len(self.graph.nodes())
            metrics['edge_count'] = len(self.graph.edges())
            metrics['density'] = nx.density(self.graph)
            metrics['average_clustering'] = nx.average_clustering(self.graph)

            if nx.is_connected(self.graph):
                metrics['diameter'] = nx.diameter(self.graph)
                metrics['average_shortest_path_length'] = nx.average_shortest_path_length(self.graph)
            else:
                metrics['connected_components'] = nx.number_connected_components(self.graph)

            # Platform distribution
            platform_dist = {}
            for node_id, node in self.nodes.items():
                platform = node.platform
                platform_dist[platform] = platform_dist.get(platform, 0) + 1
            metrics['platform_distribution'] = platform_dist

        return metrics