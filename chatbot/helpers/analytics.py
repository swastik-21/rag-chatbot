"""
Analytics module for tracking chatbot performance metrics and KPIs.
"""
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from threading import Lock
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConversationEvent:
    """Represents a single conversation event"""
    timestamp: str
    session_id: str
    event_type: str  # 'question', 'answer', 'fallback', 'error'
    question: Optional[str] = None
    answer: Optional[str] = None
    response_time_ms: Optional[float] = None
    docs_retrieved: Optional[int] = None
    sources: Optional[List[Dict]] = None
    error: Optional[str] = None
    model_used: Optional[str] = None
    answer_length: Optional[int] = None
    product_category: Optional[str] = None  # Track which product category was discussed


class AnalyticsTracker:
    """Tracks analytics metrics for the chatbot"""
    
    def __init__(self, max_events: int = 10000):
        self.max_events = max_events
        self.events: deque = deque(maxlen=max_events)
        self.lock = Lock()
        
        # Real-time metrics
        self.metrics = {
            'total_conversations': 0,
            'total_questions': 0,
            'total_answers': 0,
            'total_fallbacks': 0,
            'total_errors': 0,
            'avg_response_time_ms': 0.0,
            'total_response_time_ms': 0.0,
            'sessions': set(),
            'questions_by_hour': defaultdict(int),
            'common_questions': defaultdict(int),
            'fallback_reasons': defaultdict(int),
            'model_usage': defaultdict(int),
            'product_categories': defaultdict(int),  # Track product category mentions
        }
        
        # Session tracking
        self.session_metrics: Dict[str, Dict] = defaultdict(lambda: {
            'start_time': None,
            'question_count': 0,
            'last_activity': None,
            'total_response_time_ms': 0.0,
        })
    
    def track_event(self, event: ConversationEvent):
        """Track a conversation event"""
        with self.lock:
            self.events.append(event)
            
            # Update metrics
            if event.event_type == 'question':
                self.metrics['total_questions'] += 1
                self.metrics['sessions'].add(event.session_id)
                self.metrics['questions_by_hour'][datetime.now().strftime('%Y-%m-%d %H:00')] += 1
                
                if event.question:
                    # Track common questions (normalize)
                    normalized_q = event.question.lower().strip()[:100]
                    self.metrics['common_questions'][normalized_q] += 1
                
                # Update session metrics
                session_meta = self.session_metrics[event.session_id]
                if session_meta['start_time'] is None:
                    session_meta['start_time'] = event.timestamp
                session_meta['question_count'] += 1
                session_meta['last_activity'] = event.timestamp
            
            elif event.event_type == 'answer':
                self.metrics['total_answers'] += 1
                if event.response_time_ms:
                    self.metrics['total_response_time_ms'] += event.response_time_ms
                    session_meta = self.session_metrics[event.session_id]
                    session_meta['total_response_time_ms'] += event.response_time_ms
                    
                    # Update average
                    if self.metrics['total_answers'] > 0:
                        self.metrics['avg_response_time_ms'] = (
                            self.metrics['total_response_time_ms'] / self.metrics['total_answers']
                        )
                
                if event.model_used:
                    self.metrics['model_usage'][event.model_used] += 1
                
                # Track product category if mentioned
                if event.product_category:
                    self.metrics['product_categories'][event.product_category] += 1
                elif event.answer:
                    # Try to detect category from answer
                    answer_lower = event.answer.lower()
                    categories = {
                        'Website Agent': 'website agent' in answer_lower,
                        'Social Media Agent': 'social media agent' in answer_lower,
                        'Messenger Agent': 'messenger agent' in answer_lower,
                        'Call Agent': 'call agent' in answer_lower,
                        'GPT Store': 'gpt store' in answer_lower or 'chatgpt' in answer_lower,
                        'Electronics & Tech': 'electronics' in answer_lower or 'tech' in answer_lower,
                        'Fashion & Apparel': 'fashion' in answer_lower or 'apparel' in answer_lower,
                        'Home & Garden': 'home' in answer_lower or 'garden' in answer_lower,
                        'Agencies & Partners': 'agency' in answer_lower or 'partner' in answer_lower,
                    }
                    for category, found in categories.items():
                        if found:
                            self.metrics['product_categories'][category] += 1
                            break
            
            elif event.event_type == 'fallback':
                self.metrics['total_fallbacks'] += 1
                if event.error:
                    self.metrics['fallback_reasons'][event.error] += 1
            
            elif event.event_type == 'error':
                self.metrics['total_errors'] += 1
            
            # Update total conversations (unique sessions)
            self.metrics['total_conversations'] = len(self.metrics['sessions'])
    
    def get_kpis(self) -> Dict[str, Any]:
        """Get key performance indicators"""
        with self.lock:
            total_questions = self.metrics['total_questions']
            total_answers = self.metrics['total_answers']
            total_fallbacks = self.metrics['total_fallbacks']
            total_errors = self.metrics['total_errors']
            
            # Calculate rates
            completion_rate = (total_answers / total_questions * 100) if total_questions > 0 else 0
            fallback_rate = (total_fallbacks / total_questions * 100) if total_questions > 0 else 0
            error_rate = (total_errors / total_questions * 100) if total_questions > 0 else 0
            
            # Average questions per session
            avg_questions_per_session = (
                total_questions / len(self.metrics['sessions'])
                if len(self.metrics['sessions']) > 0 else 0
            )
            
            # First contact resolution (sessions with only 1 question)
            single_question_sessions = sum(
                1 for session_id in self.metrics['sessions']
                if self.session_metrics[session_id]['question_count'] == 1
            )
            fcr_rate = (
                single_question_sessions / len(self.metrics['sessions']) * 100
                if len(self.metrics['sessions']) > 0 else 0
            )
            
            return {
                'total_conversations': self.metrics['total_conversations'],
                'total_questions': total_questions,
                'total_answers': total_answers,
                'completion_rate': round(completion_rate, 2),
                'fallback_rate': round(fallback_rate, 2),
                'error_rate': round(error_rate, 2),
                'avg_response_time_ms': round(self.metrics['avg_response_time_ms'], 2),
                'avg_questions_per_session': round(avg_questions_per_session, 2),
                'first_contact_resolution_rate': round(fcr_rate, 2),
                'total_fallbacks': total_fallbacks,
                'total_errors': total_errors,
            }
    
    def get_top_questions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most common questions"""
        with self.lock:
            sorted_questions = sorted(
                self.metrics['common_questions'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:limit]
            
            return [
                {'question': q, 'count': count}
                for q, count in sorted_questions
            ]
    
    def get_hourly_stats(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get statistics for the last N hours"""
        with self.lock:
            now = datetime.now()
            stats = []
            
            for i in range(hours):
                hour_time = now - timedelta(hours=i)
                hour_key = hour_time.strftime('%Y-%m-%d %H:00')
                count = self.metrics['questions_by_hour'].get(hour_key, 0)
                stats.append({
                    'hour': hour_key,
                    'question_count': count
                })
            
            return list(reversed(stats))
    
    def get_model_usage_stats(self) -> Dict[str, int]:
        """Get model usage statistics"""
        with self.lock:
            return dict(self.metrics['model_usage'])
    
    def get_product_category_stats(self) -> Dict[str, int]:
        """Get product category statistics"""
        with self.lock:
            return dict(self.metrics['product_categories'])
    
    def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent events"""
        with self.lock:
            return [asdict(event) for event in list(self.events)[-limit:]]
    
    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific session"""
        with self.lock:
            if session_id not in self.session_metrics:
                return None
            
            session_meta = self.session_metrics[session_id]
            question_count = session_meta['question_count']
            total_time = session_meta['total_response_time_ms']
            
            return {
                'session_id': session_id,
                'start_time': session_meta['start_time'],
                'last_activity': session_meta['last_activity'],
                'question_count': question_count,
                'avg_response_time_ms': round(total_time / question_count, 2) if question_count > 0 else 0,
                'total_response_time_ms': round(total_time, 2),
            }
    
    def export_events(self, format: str = 'json') -> str:
        """Export events in specified format"""
        with self.lock:
            events_data = [asdict(event) for event in self.events]
            
            if format == 'json':
                return json.dumps(events_data, indent=2)
            else:
                # CSV format
                if not events_data:
                    return ""
                
                headers = list(events_data[0].keys())
                lines = [','.join(headers)]
                
                for event in events_data:
                    values = []
                    for header in headers:
                        value = event.get(header, '')
                        if isinstance(value, (list, dict)):
                            value = json.dumps(value)
                        values.append(str(value).replace(',', ';'))
                    lines.append(','.join(values))
                
                return '\n'.join(lines)


# Global analytics tracker instance
analytics_tracker = AnalyticsTracker()

