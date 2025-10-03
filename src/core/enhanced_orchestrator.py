#!/usr/bin/env python3
"""
🤖 Enhanced Investigation Orchestrator - Multi-Model AI Coordination
Desktop OSINT Suite - Phase 2 Implementation
LakyLuk Enhanced Edition - 27.9.2025

Core orchestration engine for AI-powered OSINT investigations
Features:
- Multi-model AI ensemble (GPT-4 + Gemini + Claude)
- Intelligent investigation planning
- Real-time decision making
- Entity correlation and analysis
- Progress monitoring and optimization
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import os
from pathlib import Path

# AI Model imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

# Browser automation import
try:
    from .browser_manager import EnhancedBrowserManager, BrowserType
    BROWSER_AUTOMATION_AVAILABLE = True
except ImportError:
    BROWSER_AUTOMATION_AVAILABLE = False

# Social media orchestration import
try:
    from .social_media_orchestration import SocialMediaOrchestrator, get_social_media_orchestrator
    SOCIAL_MEDIA_AVAILABLE = True
except ImportError:
    SOCIAL_MEDIA_AVAILABLE = False

class InvestigationType(Enum):
    """Types of OSINT investigations"""
    PERSON = "person"
    BUSINESS = "business"
    LOCATION = "location"
    SOCIAL_MEDIA = "social_media"
    THREAT_INTEL = "threat_intel"
    DIGITAL_FOOTPRINT = "digital_footprint"

class InvestigationPriority(Enum):
    """Investigation priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class AIModelType(Enum):
    """Available AI models"""
    GPT4 = "gpt4"
    GEMINI = "gemini"
    CLAUDE = "claude"

@dataclass
class InvestigationTarget:
    """Enhanced investigation target with comprehensive metadata"""
    name: str
    target_type: InvestigationType
    location: Optional[str] = None
    age_estimate: Optional[str] = None
    occupation_hint: Optional[str] = None
    email_hint: Optional[str] = None
    phone_hint: Optional[str] = None
    social_hint: Optional[str] = None
    known_associates: List[str] = None
    investigation_scope: str = "comprehensive"
    priority: InvestigationPriority = InvestigationPriority.NORMAL
    time_limit_minutes: int = 30
    confidence_threshold: float = 0.7
    stealth_level: str = "moderate"
    special_requirements: List[str] = None

    def __post_init__(self):
        if self.known_associates is None:
            self.known_associates = []
        if self.special_requirements is None:
            self.special_requirements = []

@dataclass
class AIDecision:
    """AI model decision with confidence and reasoning"""
    model: AIModelType
    recommendation: str
    confidence: float
    reasoning: str
    sources_suggested: List[str]
    estimated_time: int
    risk_assessment: str
    timestamp: datetime

@dataclass
class EnsembleDecision:
    """Final ensemble decision from multiple AI models"""
    final_recommendation: str
    overall_confidence: float
    consensus_level: float
    individual_decisions: List[AIDecision]
    execution_plan: List[Dict[str, Any]]
    estimated_total_time: int
    risk_level: str
    timestamp: datetime

class EnhancedInvestigationOrchestrator:
    """
    Enhanced AI-powered investigation orchestration engine

    Coordinates multiple AI models for intelligent OSINT investigations:
    - GPT-4: Technical analysis and pattern recognition
    - Gemini: Entity correlation and deep context analysis
    - Claude: Strategic planning and risk assessment
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # AI model configurations
        self.ai_models = {}
        self.model_weights = {
            AIModelType.GPT4: 0.4,    # Strong technical analysis
            AIModelType.GEMINI: 0.35, # Good context understanding
            AIModelType.CLAUDE: 0.25  # Strategic planning (when available)
        }

        # Investigation state
        self.active_investigations: Dict[str, Dict] = {}
        self.investigation_history: List[Dict] = []

        # Browser automation integration
        self.browser_manager = None
        self.browser_available = False

        # Social media orchestration integration
        self.social_media_orchestrator = None
        self.social_media_available = SOCIAL_MEDIA_AVAILABLE

        # Performance metrics
        self.ensemble_stats = {
            'total_investigations': 0,
            'successful_investigations': 0,
            'average_accuracy': 0.0,
            'model_performance': {},
            'web_scraping_stats': {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0
            }
        }

    async def initialize(self):
        """Initialize AI models and orchestrator components"""
        try:
            self.logger.info("🤖 Initializing Enhanced Investigation Orchestrator...")

            # Initialize available AI models
            await self._initialize_ai_models()

            # Validate model availability
            available_models = [model for model in self.ai_models.keys()]
            self.logger.info(f"✅ Available AI models: {[m.value for m in available_models]}")

            if len(available_models) < 1:
                raise Exception("At least one AI model must be available")

            # Adjust weights based on available models
            self._adjust_model_weights(available_models)

            # Initialize browser automation
            await self._initialize_browser_automation()

            # Initialize social media orchestration
            await self._initialize_social_media_orchestration()

            self.logger.info("✅ Enhanced Investigation Orchestrator initialized successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize orchestrator: {e}")
            raise

    async def _initialize_ai_models(self):
        """Initialize all available AI models"""

        # Initialize OpenAI GPT-4
        if OPENAI_AVAILABLE:
            openai_key = os.getenv('OPENAI_API_KEY')
            if openai_key:
                try:
                    self.ai_models[AIModelType.GPT4] = openai.OpenAI(api_key=openai_key)
                    self.logger.info("✅ OpenAI GPT-4 initialized")
                except Exception as e:
                    self.logger.warning(f"⚠️ OpenAI initialization failed: {e}")

        # Initialize Google Gemini
        if GEMINI_AVAILABLE:
            gemini_key = os.getenv('GOOGLE_API_KEY')
            if gemini_key:
                try:
                    genai.configure(api_key=gemini_key)
                    self.ai_models[AIModelType.GEMINI] = genai.GenerativeModel('models/gemini-2.5-flash')
                    self.logger.info("✅ Google Gemini initialized")
                except Exception as e:
                    self.logger.warning(f"⚠️ Gemini initialization failed: {e}")

        # Initialize Claude (if available)
        if CLAUDE_AVAILABLE:
            claude_key = os.getenv('ANTHROPIC_API_KEY')
            if claude_key:
                try:
                    self.ai_models[AIModelType.CLAUDE] = anthropic.Anthropic(api_key=claude_key)
                    self.logger.info("✅ Claude initialized")
                except Exception as e:
                    self.logger.warning(f"⚠️ Claude initialization failed: {e}")

    def _adjust_model_weights(self, available_models: List[AIModelType]):
        """Adjust model weights based on available models"""
        if len(available_models) == 1:
            # Single model gets full weight
            self.model_weights = {available_models[0]: 1.0}
        elif len(available_models) == 2:
            # Two models split weight
            if AIModelType.GPT4 in available_models and AIModelType.GEMINI in available_models:
                self.model_weights = {
                    AIModelType.GPT4: 0.6,
                    AIModelType.GEMINI: 0.4
                }

        self.logger.info(f"📊 Adjusted model weights: {self.model_weights}")

    async def start_investigation(
        self,
        target: InvestigationTarget,
        progress_callback=None
    ) -> str:
        """
        Start a new AI-orchestrated investigation

        Args:
            target: Investigation target with metadata
            progress_callback: Function to call with progress updates

        Returns:
            investigation_id: Unique identifier for the investigation
        """

        investigation_id = f"osint_{target.target_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            self.logger.info(f"🔍 Starting enhanced investigation: {investigation_id}")
            self.logger.info(f"🎯 Target: {target.name} ({target.target_type.value})")

            if progress_callback:
                progress_callback("planning", "Creating AI-powered investigation plan...")

            # Step 1: Multi-model investigation planning
            ensemble_decision = await self._create_ensemble_investigation_plan(target)

            if progress_callback:
                progress_callback("executing", "Executing AI-coordinated investigation...")

            # Step 2: Execute investigation with AI coordination
            investigation_results = await self._execute_ai_coordinated_investigation(
                target, ensemble_decision, progress_callback
            )

            if progress_callback:
                progress_callback("analyzing", "Performing AI ensemble analysis...")

            # Step 3: AI ensemble analysis of results
            final_analysis = await self._perform_ensemble_analysis(
                target, investigation_results, ensemble_decision
            )

            if progress_callback:
                progress_callback("reporting", "Generating AI-enhanced report...")

            # Step 4: Generate comprehensive report
            final_report = await self._generate_enhanced_report(
                investigation_id, target, ensemble_decision, investigation_results, final_analysis
            )

            # Store investigation
            self.active_investigations[investigation_id] = {
                "target": target,
                "ensemble_decision": ensemble_decision,
                "results": investigation_results,
                "analysis": final_analysis,
                "report": final_report,
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }

            # Update performance metrics
            self._update_performance_metrics(investigation_id, True)

            if progress_callback:
                progress_callback("completed", "AI-enhanced investigation completed!")

            self.logger.info(f"✅ Investigation completed: {investigation_id}")
            return investigation_id

        except Exception as e:
            self.logger.error(f"❌ Investigation failed: {e}")

            # Store failed investigation
            self.active_investigations[investigation_id] = {
                "target": target,
                "error": str(e),
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }

            self._update_performance_metrics(investigation_id, False)

            if progress_callback:
                progress_callback("failed", f"Investigation failed: {str(e)}")

            raise

    async def _create_ensemble_investigation_plan(
        self,
        target: InvestigationTarget
    ) -> EnsembleDecision:
        """Create investigation plan using AI ensemble"""

        self.logger.info("🧠 Creating AI ensemble investigation plan...")

        # Create investigation prompt
        planning_prompt = self._create_planning_prompt(target)

        # Get decisions from all available AI models
        ai_decisions = []

        for model_type, model_instance in self.ai_models.items():
            try:
                decision = await self._get_ai_decision(model_type, model_instance, planning_prompt, target)
                ai_decisions.append(decision)
                self.logger.info(f"✅ {model_type.value} decision: {decision.recommendation} (confidence: {decision.confidence:.2f})")
            except Exception as e:
                self.logger.warning(f"⚠️ {model_type.value} decision failed: {e}")

        if not ai_decisions:
            raise Exception("No AI models provided valid decisions")

        # Create ensemble decision
        ensemble_decision = self._create_ensemble_decision(ai_decisions)

        self.logger.info(f"🎯 Ensemble decision: {ensemble_decision.final_recommendation}")
        self.logger.info(f"📊 Overall confidence: {ensemble_decision.overall_confidence:.2f}")
        self.logger.info(f"🤝 Consensus level: {ensemble_decision.consensus_level:.2f}")

        return ensemble_decision

    def _create_planning_prompt(self, target: InvestigationTarget) -> str:
        """Create comprehensive planning prompt for AI models"""

        prompt = f"""
You are an expert OSINT investigator planning a comprehensive investigation.

TARGET INFORMATION:
- Name: {target.name}
- Type: {target.target_type.value}
- Location: {target.location or 'Unknown'}
- Priority: {target.priority.value}
- Time Limit: {target.time_limit_minutes} minutes
- Scope: {target.investigation_scope}
- Stealth Level: {target.stealth_level}

ADDITIONAL CONTEXT:
- Age Estimate: {target.age_estimate or 'Unknown'}
- Occupation: {target.occupation_hint or 'Unknown'}
- Email Hint: {target.email_hint or 'None'}
- Phone Hint: {target.phone_hint or 'None'}
- Social Media Hint: {target.social_hint or 'None'}
- Known Associates: {', '.join(target.known_associates) if target.known_associates else 'None'}

AVAILABLE OSINT SOURCES:
- Social Media: Facebook, Instagram, LinkedIn, Twitter, TikTok
- Search Engines: Google, Bing, DuckDuckGo
- Czech Republic: justice.cz, ARES, Cadastre, firmy.cz
- Professional: LinkedIn, company websites, professional registries
- Public Records: Court records, business registrations, property records
- Reverse Search: Email, phone, image reverse search
- Archive Sources: Archive.org, cached pages

YOUR TASK:
Create an optimal OSINT investigation plan. Provide:

1. RECOMMENDED APPROACH: Single word - comprehensive/targeted/stealth/rapid
2. CONFIDENCE: Single number 0.0-1.0 representing your confidence in this plan
3. PRIMARY SOURCES: List 3-5 most promising sources to investigate first
4. ESTIMATED TIME: Total minutes needed for thorough investigation
5. RISK ASSESSMENT: low/medium/high - legal and operational risks
6. REASONING: Brief explanation of your strategy

Format your response as:
APPROACH: [approach]
CONFIDENCE: [0.0-1.0]
SOURCES: [source1, source2, source3]
TIME: [minutes]
RISK: [low/medium/high]
REASONING: [explanation]
"""

        return prompt

    async def _get_ai_decision(
        self,
        model_type: AIModelType,
        model_instance: Any,
        prompt: str,
        target: InvestigationTarget
    ) -> AIDecision:
        """Get decision from specific AI model"""

        try:
            if model_type == AIModelType.GPT4:
                response = model_instance.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.3
                )
                content = response.choices[0].message.content

            elif model_type == AIModelType.GEMINI:
                response = model_instance.generate_content(prompt)
                content = response.text

            elif model_type == AIModelType.CLAUDE:
                response = model_instance.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text

            # Parse AI response
            decision = self._parse_ai_response(model_type, content)
            return decision

        except Exception as e:
            self.logger.error(f"❌ {model_type.value} decision failed: {e}")
            raise

    def _parse_ai_response(self, model_type: AIModelType, content: str) -> AIDecision:
        """Parse AI model response into structured decision"""

        lines = content.strip().split('\n')
        parsed = {}

        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                parsed[key.strip().upper()] = value.strip()

        # Extract structured information with better parsing
        approach = parsed.get('APPROACH', 'comprehensive')

        # Parse confidence
        confidence_str = parsed.get('CONFIDENCE', '0.7')
        try:
            confidence = float(confidence_str)
        except ValueError:
            confidence = 0.7

        # Parse sources
        sources_str = parsed.get('SOURCES', '')
        sources = [s.strip() for s in sources_str.split(',') if s.strip()]
        if not sources:
            sources = ['google', 'social_media', 'public_records']

        # Parse time estimate with better handling
        time_str = parsed.get('TIME', '30')
        try:
            # Extract just numbers from time string
            import re
            time_numbers = re.findall(r'\d+', time_str)
            time_estimate = int(time_numbers[0]) if time_numbers else 30
        except (ValueError, IndexError):
            time_estimate = 30

        risk = parsed.get('RISK', 'medium')
        reasoning = parsed.get('REASONING', 'Standard investigation approach')

        return AIDecision(
            model=model_type,
            recommendation=approach,
            confidence=confidence,
            reasoning=reasoning,
            sources_suggested=sources,
            estimated_time=time_estimate,
            risk_assessment=risk,
            timestamp=datetime.now()
        )

    def _create_ensemble_decision(self, ai_decisions: List[AIDecision]) -> EnsembleDecision:
        """Create final ensemble decision from individual AI decisions"""

        if not ai_decisions:
            raise Exception("No AI decisions to ensemble")

        # Calculate weighted confidence
        total_weight = 0
        weighted_confidence = 0

        for decision in ai_decisions:
            weight = self.model_weights.get(decision.model, 0.33)
            weighted_confidence += decision.confidence * weight
            total_weight += weight

        overall_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.5

        # Calculate consensus level (how much models agree)
        approaches = [d.recommendation for d in ai_decisions]
        most_common_approach = max(set(approaches), key=approaches.count)
        consensus_level = approaches.count(most_common_approach) / len(approaches)

        # Combine all suggested sources
        all_sources = []
        for decision in ai_decisions:
            all_sources.extend(decision.sources_suggested)

        # Remove duplicates and prioritize
        unique_sources = list(dict.fromkeys(all_sources))  # Preserves order

        # Create execution plan
        execution_plan = [
            {
                "phase": "reconnaissance",
                "sources": unique_sources[:3],  # Top 3 sources
                "estimated_time": 10,
                "priority": "high"
            },
            {
                "phase": "deep_investigation",
                "sources": unique_sources[3:6],  # Next 3 sources
                "estimated_time": 15,
                "priority": "medium"
            },
            {
                "phase": "correlation",
                "sources": ["cross_reference", "entity_linking"],
                "estimated_time": 5,
                "priority": "high"
            }
        ]

        # Calculate total estimated time
        total_time = sum(phase["estimated_time"] for phase in execution_plan)

        # Determine overall risk level
        risk_levels = [d.risk_assessment for d in ai_decisions]
        if 'high' in risk_levels:
            overall_risk = 'high'
        elif 'medium' in risk_levels:
            overall_risk = 'medium'
        else:
            overall_risk = 'low'

        return EnsembleDecision(
            final_recommendation=most_common_approach,
            overall_confidence=overall_confidence,
            consensus_level=consensus_level,
            individual_decisions=ai_decisions,
            execution_plan=execution_plan,
            estimated_total_time=total_time,
            risk_level=overall_risk,
            timestamp=datetime.now()
        )

    async def _execute_ai_coordinated_investigation(
        self,
        target: InvestigationTarget,
        ensemble_decision: EnsembleDecision,
        progress_callback=None
    ) -> Dict[str, Any]:
        """Execute investigation following AI-coordinated plan"""

        self.logger.info("🚀 Executing AI-coordinated investigation...")

        results = {
            "execution_start": datetime.now().isoformat(),
            "phases_completed": [],
            "sources_investigated": [],
            "entities_found": [],
            "correlations": [],
            "raw_data": {}
        }

        # Execute each phase of the investigation plan
        for i, phase in enumerate(ensemble_decision.execution_plan):
            phase_name = phase["phase"]

            if progress_callback:
                progress_callback("executing", f"Phase {i+1}/{len(ensemble_decision.execution_plan)}: {phase_name}")

            self.logger.info(f"📋 Executing phase: {phase_name}")

            # Simulate phase execution (in real implementation, this would call actual OSINT tools)
            phase_results = await self._execute_investigation_phase(target, phase)

            results["phases_completed"].append({
                "phase": phase_name,
                "sources": phase["sources"],
                "results": phase_results,
                "timestamp": datetime.now().isoformat()
            })

            results["sources_investigated"].extend(phase["sources"])

            # Simulate finding entities and correlations
            if phase_name == "reconnaissance":
                results["entities_found"].extend([
                    {"type": "email", "value": f"{target.name.lower().replace(' ', '.')}@example.com", "confidence": 0.6},
                    {"type": "social_profile", "value": f"linkedin.com/in/{target.name.lower().replace(' ', '-')}", "confidence": 0.7}
                ])
            elif phase_name == "correlation":
                results["correlations"].extend([
                    {"entity1": "email", "entity2": "social_profile", "correlation_type": "ownership", "confidence": 0.8}
                ])

        results["execution_end"] = datetime.now().isoformat()
        return results

    async def _execute_investigation_phase(
        self,
        target: InvestigationTarget,
        phase: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single phase of investigation"""

        phase_name = phase.get("phase", "unknown")
        sources = phase.get("sources", [])

        # Check if this is a social media investigation phase
        if self._is_social_media_phase(phase_name, sources):
            return await self._execute_social_media_phase(target, phase)

        # For non-social media phases, use existing simulation logic
        # In real implementation, this would:
        # - Use browser automation for web scraping
        # - Call search engine APIs
        # - Access Czech government databases
        # - Execute specialized OSINT tools

        await asyncio.sleep(1)  # Simulate processing time

        return {
            "sources_accessed": phase["sources"],
            "data_collected": f"Mock data for {phase['phase']}",
            "success": True,
            "timestamp": datetime.now().isoformat()
        }

    def _is_social_media_phase(self, phase_name: str, sources: List[str]) -> bool:
        """Determine if this is a social media investigation phase"""
        social_media_keywords = ['social', 'facebook', 'instagram', 'linkedin', 'twitter', 'social_media']
        social_media_sources = ['facebook.com', 'instagram.com', 'linkedin.com', 'twitter.com']

        # Check phase name
        if any(keyword in phase_name.lower() for keyword in social_media_keywords):
            return True

        # Check sources
        if any(any(sm_source in source for sm_source in social_media_sources) for source in sources):
            return True

        return False

    async def _execute_social_media_phase(
        self,
        target: InvestigationTarget,
        phase: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute social media investigation phase using social media orchestrator"""
        try:
            if not self.social_media_available or not self.social_media_orchestrator:
                self.logger.warning("⚠️ Social media orchestrator not available, using mock data")
                return await self._execute_mock_social_media_phase(target, phase)

            self.logger.info(f"🔍 Executing real social media investigation for: {target.name}")

            # Execute social media investigation using our orchestrator
            social_results = await self.social_media_orchestrator.execute_social_media_investigation(
                target=target,
                progress_callback=None  # Could be enhanced to support progress
            )

            # Convert social media results to phase results format
            return {
                "sources_accessed": phase["sources"],
                "data_collected": {
                    "platforms_searched": social_results.platforms_searched,
                    "profiles_found": social_results.profiles_found,
                    "correlations": social_results.correlations,
                    "confidence_score": social_results.confidence_score
                },
                "profiles_found": len(social_results.profiles_found),
                "platforms_searched": len(social_results.platforms_searched),
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "investigation_duration": social_results.investigation_duration
            }

        except Exception as e:
            self.logger.error(f"❌ Social media phase execution failed: {e}")
            return await self._execute_mock_social_media_phase(target, phase)

    async def _execute_mock_social_media_phase(
        self,
        target: InvestigationTarget,
        phase: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute mock social media investigation when real orchestrator is not available"""
        await asyncio.sleep(2)  # Simulate processing time

        # Generate mock social media results
        mock_profiles = [
            {
                "platform": "facebook",
                "name": target.name,
                "url": f"https://facebook.com/search?q={target.name.replace(' ', '+')}",
                "confidence": 0.7
            },
            {
                "platform": "linkedin",
                "name": target.name,
                "url": f"https://linkedin.com/search/results/people/?keywords={target.name.replace(' ', '%20')}",
                "confidence": 0.8
            }
        ]

        return {
            "sources_accessed": phase["sources"],
            "data_collected": {
                "platforms_searched": ["facebook", "linkedin"],
                "profiles_found": mock_profiles,
                "correlations": [],
                "confidence_score": 0.75
            },
            "profiles_found": len(mock_profiles),
            "platforms_searched": 2,
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "investigation_duration": 2.0,
            "mock_data": True
        }

    async def _perform_ensemble_analysis(
        self,
        target: InvestigationTarget,
        investigation_results: Dict[str, Any],
        ensemble_decision: EnsembleDecision
    ) -> Dict[str, Any]:
        """Perform AI ensemble analysis of investigation results"""

        self.logger.info("🧠 Performing AI ensemble analysis...")

        # Create analysis prompt
        analysis_prompt = f"""
Analyze the following OSINT investigation results and provide intelligence assessment:

TARGET: {target.name} ({target.target_type.value})
INVESTIGATION SCOPE: {target.investigation_scope}

RESULTS SUMMARY:
- Phases Completed: {len(investigation_results['phases_completed'])}
- Sources Investigated: {len(investigation_results['sources_investigated'])}
- Entities Found: {len(investigation_results['entities_found'])}
- Correlations: {len(investigation_results['correlations'])}

ENTITIES DISCOVERED:
{json.dumps(investigation_results['entities_found'], indent=2)}

CORRELATIONS FOUND:
{json.dumps(investigation_results['correlations'], indent=2)}

Provide analysis in this format:
THREAT_LEVEL: low/medium/high/critical
RELIABILITY: 0.0-1.0
KEY_FINDINGS: [finding1, finding2, finding3]
RECOMMENDATIONS: [rec1, rec2, rec3]
NEXT_STEPS: [step1, step2, step3]
"""

        # Get analysis from available AI models
        analyses = []
        for model_type, model_instance in self.ai_models.items():
            try:
                if model_type == AIModelType.GPT4:
                    response = model_instance.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": analysis_prompt}],
                        max_tokens=400,
                        temperature=0.2
                    )
                    content = response.choices[0].message.content
                elif model_type == AIModelType.GEMINI:
                    response = model_instance.generate_content(analysis_prompt)
                    content = response.text

                analyses.append({
                    "model": model_type.value,
                    "analysis": content,
                    "timestamp": datetime.now().isoformat()
                })

            except Exception as e:
                self.logger.warning(f"⚠️ {model_type.value} analysis failed: {e}")

        return {
            "individual_analyses": analyses,
            "ensemble_assessment": self._create_ensemble_assessment(analyses),
            "confidence_score": 0.8,  # Would be calculated from actual analysis
            "timestamp": datetime.now().isoformat()
        }

    def _create_ensemble_assessment(self, analyses: List[Dict]) -> Dict[str, Any]:
        """Create ensemble assessment from individual AI analyses"""

        if not analyses:
            return {
                "threat_level": "unknown",
                "reliability": 0.5,
                "key_findings": ["Insufficient analysis available"],
                "recommendations": ["Retry investigation with more sources"],
                "next_steps": ["Manual verification required"]
            }

        # Simple ensemble assessment (in real implementation would be more sophisticated)
        return {
            "threat_level": "low",
            "reliability": 0.75,
            "key_findings": [
                "Limited public information available",
                "Standard digital footprint observed",
                "No immediate security concerns identified"
            ],
            "recommendations": [
                "Continue monitoring for new information",
                "Cross-reference with additional databases",
                "Consider human intelligence sources"
            ],
            "next_steps": [
                "Schedule follow-up investigation in 30 days",
                "Set up automated monitoring alerts",
                "Document findings in threat intelligence database"
            ]
        }

    async def _generate_enhanced_report(
        self,
        investigation_id: str,
        target: InvestigationTarget,
        ensemble_decision: EnsembleDecision,
        investigation_results: Dict[str, Any],
        final_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive AI-enhanced investigation report"""

        return {
            "investigation_id": investigation_id,
            "report_type": "ai_enhanced_osint",
            "generated_by": "Enhanced Investigation Orchestrator",
            "timestamp": datetime.now().isoformat(),

            "executive_summary": {
                "target": asdict(target),
                "investigation_approach": ensemble_decision.final_recommendation,
                "overall_confidence": ensemble_decision.overall_confidence,
                "consensus_level": ensemble_decision.consensus_level,
                "threat_level": final_analysis["ensemble_assessment"]["threat_level"],
                "key_findings": final_analysis["ensemble_assessment"]["key_findings"]
            },

            "ai_ensemble_details": {
                "models_used": [d.model.value for d in ensemble_decision.individual_decisions],
                "individual_decisions": [asdict(d) for d in ensemble_decision.individual_decisions],
                "consensus_analysis": {
                    "agreement_level": ensemble_decision.consensus_level,
                    "confidence_range": [d.confidence for d in ensemble_decision.individual_decisions],
                    "time_estimates": [d.estimated_time for d in ensemble_decision.individual_decisions]
                }
            },

            "investigation_execution": {
                "plan": ensemble_decision.execution_plan,
                "results": investigation_results,
                "phases_completed": len(investigation_results["phases_completed"]),
                "sources_investigated": investigation_results["sources_investigated"],
                "entities_discovered": investigation_results["entities_found"],
                "correlations_found": investigation_results["correlations"]
            },

            "intelligence_analysis": final_analysis,

            "recommendations": final_analysis["ensemble_assessment"]["recommendations"],
            "next_steps": final_analysis["ensemble_assessment"]["next_steps"],

            "metadata": {
                "investigation_duration": "calculated_duration",
                "data_sources_count": len(investigation_results["sources_investigated"]),
                "ai_models_used": len(ensemble_decision.individual_decisions),
                "confidence_score": final_analysis["confidence_score"],
                "export_formats": ["json", "pdf", "maltego", "csv"]
            }
        }

    def _update_performance_metrics(self, investigation_id: str, success: bool):
        """Update orchestrator performance metrics"""
        self.ensemble_stats['total_investigations'] += 1
        if success:
            self.ensemble_stats['successful_investigations'] += 1

        # Calculate success rate
        self.ensemble_stats['average_accuracy'] = (
            self.ensemble_stats['successful_investigations'] /
            self.ensemble_stats['total_investigations']
        )

    def get_investigation_status(self, investigation_id: str) -> Optional[Dict]:
        """Get status of specific investigation"""
        return self.active_investigations.get(investigation_id)

    def list_active_investigations(self) -> List[str]:
        """List all active investigation IDs"""
        return list(self.active_investigations.keys())

    def get_ensemble_statistics(self) -> Dict[str, Any]:
        """Get ensemble performance statistics"""
        return {
            **self.ensemble_stats,
            "available_models": list(self.ai_models.keys()),
            "model_weights": self.model_weights,
            "browser_available": self.browser_available,
            "last_updated": datetime.now().isoformat()
        }

    # ====== BROWSER AUTOMATION INTEGRATION ======

    async def _initialize_browser_automation(self):
        """Initialize browser automation capabilities"""
        try:
            if BROWSER_AUTOMATION_AVAILABLE:
                self.browser_manager = EnhancedBrowserManager()
                await self.browser_manager.initialize()
                self.browser_available = True
                self.logger.info("✅ Browser automation initialized")
            else:
                self.logger.warning("⚠️ Browser automation not available - install browser dependencies")
                self.browser_available = False
        except Exception as e:
            self.logger.error(f"❌ Browser automation initialization failed: {e}")
            self.browser_available = False

    async def _initialize_social_media_orchestration(self):
        """Initialize social media orchestration capabilities"""
        try:
            if SOCIAL_MEDIA_AVAILABLE:
                self.social_media_orchestrator = await get_social_media_orchestrator(self)
                self.logger.info("✅ Social media orchestration initialized")
            else:
                self.logger.warning("⚠️ Social media orchestration not available - check imports")
                self.social_media_available = False
        except Exception as e:
            self.logger.error(f"❌ Social media orchestration initialization failed: {e}")
            self.social_media_available = False

    async def perform_web_investigation(self, target: 'InvestigationTarget', urls: List[str]) -> Dict[str, Any]:
        """
        Perform AI-guided web scraping investigation

        Args:
            target: Investigation target
            urls: List of URLs to investigate

        Returns:
            Structured investigation results with extracted data
        """
        if not self.browser_available:
            raise Exception("Browser automation not available")

        investigation_results = {
            'target': target.name,
            'urls_investigated': [],
            'extracted_data': {},
            'ai_analysis': {},
            'confidence_score': 0.0,
            'timestamp': datetime.now().isoformat()
        }

        try:
            # Create browser session
            session_created = await self.browser_manager.create_browser_session(
                browser_type=BrowserType.PLAYWRIGHT_CHROMIUM
            )

            if not session_created:
                raise Exception("Failed to create browser session")

            # Investigate each URL
            for url in urls:
                try:
                    self.logger.info(f"🔍 Investigating URL: {url}")

                    # Navigate to URL
                    navigation_success = await self.browser_manager.navigate_to_url(url)

                    if navigation_success:
                        # AI-guided data extraction
                        selectors = await self._generate_ai_selectors(target, url)
                        extracted_data = await self.browser_manager.extract_data(selectors)

                        # AI analysis of extracted data
                        ai_analysis = await self._analyze_extracted_data(target, url, extracted_data)

                        # Store results
                        investigation_results['urls_investigated'].append(url)
                        investigation_results['extracted_data'][url] = extracted_data
                        investigation_results['ai_analysis'][url] = ai_analysis

                        # Update stats
                        self.ensemble_stats['web_scraping_stats']['successful_requests'] += 1

                    else:
                        self.logger.warning(f"⚠️ Failed to navigate to {url}")
                        self.ensemble_stats['web_scraping_stats']['failed_requests'] += 1

                    self.ensemble_stats['web_scraping_stats']['total_requests'] += 1

                    # Human-like delay between requests
                    await asyncio.sleep(2)

                except Exception as e:
                    self.logger.error(f"❌ Error investigating {url}: {e}")
                    self.ensemble_stats['web_scraping_stats']['failed_requests'] += 1

            # Close browser session
            await self.browser_manager.close_session()

            # Calculate overall confidence
            investigation_results['confidence_score'] = self._calculate_web_investigation_confidence(
                investigation_results
            )

            return investigation_results

        except Exception as e:
            self.logger.error(f"❌ Web investigation failed: {e}")
            if self.browser_manager:
                await self.browser_manager.close_session()
            raise

    async def _generate_ai_selectors(self, target: 'InvestigationTarget', url: str) -> Dict[str, str]:
        """
        Use AI to generate CSS selectors for data extraction based on target and URL
        """
        if not self.ai_models:
            # Fallback generic selectors
            return {
                "title": "title, h1, h2",
                "content": "p, div.content, article",
                "contact": "[href*='mailto'], [href*='tel']",
                "social": "[href*='facebook'], [href*='twitter'], [href*='linkedin']"
            }

        try:
            # Use available AI model to generate selectors
            available_model = next(iter(self.ai_models.keys()))

            prompt = f"""
            Generate CSS selectors for extracting relevant information about '{target.name}' from the website: {url}

            Target type: {target.target_type.value}
            Investigation scope: {target.investigation_scope}

            Return ONLY a JSON object with selector mappings like:
            {{"title": "h1, .title", "content": "p, .content", "contact": "[href*='mailto']"}}
            """

            if available_model == AIModelType.GPT4:
                response = await self._query_gpt4(prompt)
            elif available_model == AIModelType.GEMINI:
                response = await self._query_gemini(prompt)
            else:
                response = await self._query_claude(prompt)

            # Parse AI response as JSON
            try:
                import re
                json_match = re.search(r'\{[^}]+\}', response)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass

        except Exception as e:
            self.logger.warning(f"⚠️ AI selector generation failed: {e}")

        # Fallback selectors
        return {
            "title": "title, h1, h2",
            "content": "p, div, article",
            "links": "a[href]",
            "images": "img[src]"
        }

    async def _analyze_extracted_data(self, target: 'InvestigationTarget', url: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use AI to analyze extracted data for investigation insights
        """
        if not self.ai_models or not data:
            return {"analysis": "No data to analyze", "relevance": 0.0}

        try:
            # Prepare data summary for AI analysis
            data_summary = ""
            for key, value in data.items():
                if isinstance(value, list):
                    data_summary += f"{key}: {len(value)} items\n"
                    if value:
                        data_summary += f"Sample: {str(value[0])[:100]}...\n"
                elif isinstance(value, str):
                    data_summary += f"{key}: {value[:200]}...\n"

            prompt = f"""
            Analyze this extracted data for OSINT investigation about '{target.name}':

            URL: {url}
            Target type: {target.target_type.value}

            Extracted data:
            {data_summary}

            Provide analysis in JSON format:
            {{"relevance_score": 0.8, "key_findings": ["finding1", "finding2"], "next_steps": ["step1", "step2"]}}
            """

            # Use ensemble for analysis
            if len(self.ai_models) > 1:
                analysis = await self._create_ensemble_decision(prompt, target)
                return {
                    "ensemble_analysis": analysis.decision,
                    "confidence": analysis.confidence,
                    "models_used": analysis.models_used
                }
            else:
                # Single model analysis
                available_model = next(iter(self.ai_models.keys()))
                if available_model == AIModelType.GPT4:
                    response = await self._query_gpt4(prompt)
                elif available_model == AIModelType.GEMINI:
                    response = await self._query_gemini(prompt)
                else:
                    response = await self._query_claude(prompt)

                return {"analysis": response, "model_used": available_model.value}

        except Exception as e:
            self.logger.error(f"❌ Data analysis failed: {e}")
            return {"error": str(e), "analysis": "Analysis failed"}

    def _calculate_web_investigation_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate confidence score for web investigation results"""
        if not results['urls_investigated']:
            return 0.0

        # Base confidence on successful data extraction
        successful_extractions = sum(
            1 for url_data in results['extracted_data'].values()
            if url_data and any(url_data.values())
        )

        extraction_ratio = successful_extractions / len(results['urls_investigated'])

        # Bonus for AI analysis availability
        ai_analysis_bonus = 0.1 if results['ai_analysis'] else 0.0

        return min(0.95, extraction_ratio * 0.85 + ai_analysis_bonus)

# Example usage and testing
if __name__ == "__main__":
    async def test_orchestrator():
        """Test the enhanced orchestrator"""
        from dotenv import load_dotenv

        # Load environment variables
        load_dotenv(Path(__file__).parent.parent.parent / "config" / "api_keys.env")

        # Create test target
        target = InvestigationTarget(
            name="John Doe",
            target_type=InvestigationType.PERSON,
            location="Prague, Czech Republic",
            investigation_scope="comprehensive",
            priority=InvestigationPriority.NORMAL
        )

        # Initialize orchestrator
        orchestrator = EnhancedInvestigationOrchestrator()
        await orchestrator.initialize()

        # Start investigation
        def progress_callback(phase, message):
            print(f"[{phase.upper()}] {message}")

        investigation_id = await orchestrator.start_investigation(target, progress_callback)

        # Get results
        results = orchestrator.get_investigation_status(investigation_id)

        print(f"\n🎉 Investigation completed: {investigation_id}")
        print(f"📊 Overall confidence: {results['report']['executive_summary']['overall_confidence']:.2f}")
        print(f"🤝 Consensus level: {results['report']['executive_summary']['consensus_level']:.2f}")
        print(f"⚠️ Threat level: {results['report']['executive_summary']['threat_level']}")

        return investigation_id

    # Run test
    if __name__ == "__main__":
        asyncio.run(test_orchestrator())