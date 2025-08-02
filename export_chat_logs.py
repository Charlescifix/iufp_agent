#!/usr/bin/env python3
"""
Chat Logs Export and Analysis Tool
==================================

Export chat conversation logs for RAG training and analysis.
Supports multiple output formats: JSON, CSV, Excel.

Usage:
    python export_chat_logs.py --format json --output logs.json
    python export_chat_logs.py --format csv --days 7
    python export_chat_logs.py --format excel --analysis
"""

import asyncio
import argparse
import json
import csv
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.vectorstore import PostgreSQLVectorStore
from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)


class ChatLogExporter:
    def __init__(self):
        self.vector_store = PostgreSQLVectorStore()
    
    async def get_chat_logs(self, 
                           days_back: Optional[int] = None,
                           session_id: Optional[str] = None,
                           limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve chat logs from database with optional filters.
        
        Args:
            days_back: Number of days to look back (None for all)
            session_id: Specific session ID to filter by
            limit: Maximum number of messages to return
        """
        logger.info("Retrieving chat logs", days_back=days_back, session_id=session_id, limit=limit)
        
        with self.vector_store.SessionLocal() as session:
            from src.vectorstore import ChatMessageEntity
            
            query = session.query(ChatMessageEntity).order_by(
                ChatMessageEntity.created_at.desc()
            )
            
            # Apply filters
            if days_back:
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                query = query.filter(ChatMessageEntity.created_at >= cutoff_date.isoformat())
            
            if session_id:
                query = query.filter(ChatMessageEntity.session_id == session_id)
            
            if limit:
                query = query.limit(limit)
            
            messages = query.all()
            
            # Convert to dictionaries
            chat_logs = []
            for msg in messages:
                log_entry = {
                    'message_id': msg.message_id,
                    'session_id': msg.session_id,
                    'user_message': msg.user_message,
                    'bot_response': msg.bot_response,
                    'sources': json.loads(msg.sources) if msg.sources else [],
                    'created_at': msg.created_at,
                    'processing_time': msg.processing_time,
                    'user_ip': msg.user_ip,
                    # Analysis fields
                    'user_message_length': len(msg.user_message),
                    'bot_response_length': len(msg.bot_response),
                    'sources_count': len(json.loads(msg.sources) if msg.sources else []),
                    'response_quality': 'good' if msg.processing_time < 5.0 else 'slow'
                }
                chat_logs.append(log_entry)
            
            logger.info(f"Retrieved {len(chat_logs)} chat messages")
            return chat_logs
    
    async def export_to_json(self, logs: List[Dict], output_file: str) -> None:
        """Export logs to JSON format."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'exported_at': datetime.utcnow().isoformat(),
                'total_messages': len(logs),
                'messages': logs
            }, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Exported {len(logs)} messages to {output_file}")
    
    async def export_to_csv(self, logs: List[Dict], output_file: str) -> None:
        """Export logs to CSV format."""
        if not logs:
            logger.warning("No logs to export")
            return
        
        # Flatten sources array to string
        for log in logs:
            log['sources_list'] = ', '.join(log['sources']) if log['sources'] else ''
        
        fieldnames = [
            'message_id', 'session_id', 'created_at', 'user_ip',
            'user_message', 'bot_response', 'sources_list', 'sources_count',
            'processing_time', 'user_message_length', 'bot_response_length',
            'response_quality'
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for log in logs:
                # Write only the fields we want in CSV
                row = {field: log.get(field, '') for field in fieldnames}
                writer.writerow(row)
        
        logger.info(f"Exported {len(logs)} messages to {output_file}")
    
    async def export_to_excel(self, logs: List[Dict], output_file: str, include_analysis: bool = False) -> None:
        """Export logs to Excel format with optional analysis sheets."""
        if not logs:
            logger.warning("No logs to export")
            return
        
        # Prepare data for Excel
        df_logs = pd.DataFrame(logs)
        
        # Clean up sources column for Excel
        df_logs['sources_list'] = df_logs['sources'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else ''
        )
        df_logs = df_logs.drop('sources', axis=1)
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Main logs sheet
            df_logs.to_excel(writer, sheet_name='Chat Logs', index=False)
            
            if include_analysis:
                # Analysis sheets
                await self._add_analysis_sheets(writer, df_logs)
        
        logger.info(f"Exported {len(logs)} messages to {output_file}")
    
    async def _add_analysis_sheets(self, writer, df_logs: pd.DataFrame) -> None:
        """Add analysis sheets to Excel workbook."""
        
        # 1. Session Analysis
        session_stats = df_logs.groupby('session_id').agg({
            'message_id': 'count',
            'processing_time': ['mean', 'max'],
            'user_message_length': 'mean',
            'bot_response_length': 'mean',
            'sources_count': 'mean',
            'created_at': ['min', 'max']
        }).round(2)
        session_stats.columns = ['message_count', 'avg_processing_time', 'max_processing_time',
                                'avg_user_msg_length', 'avg_bot_response_length', 'avg_sources_count',
                                'first_message', 'last_message']
        session_stats.to_excel(writer, sheet_name='Session Analysis')
        
        # 2. Daily Activity
        df_logs['date'] = pd.to_datetime(df_logs['created_at']).dt.date
        daily_stats = df_logs.groupby('date').agg({
            'message_id': 'count',
            'session_id': 'nunique',
            'processing_time': 'mean',
            'sources_count': 'mean'
        }).round(2)
        daily_stats.columns = ['messages_count', 'unique_sessions', 'avg_processing_time', 'avg_sources_count']
        daily_stats.to_excel(writer, sheet_name='Daily Activity')
        
        # 3. Common Questions (by user message similarity)
        user_messages = df_logs[['user_message', 'user_message_length']].copy()
        user_messages = user_messages.sort_values('user_message_length', ascending=False)
        user_messages.to_excel(writer, sheet_name='User Questions', index=False)
        
        # 4. Performance Analysis
        perf_stats = pd.DataFrame({
            'Metric': ['Total Messages', 'Unique Sessions', 'Avg Processing Time', 
                      'Messages with Sources', 'Avg Response Length', 'Slow Responses (>5s)'],
            'Value': [
                len(df_logs),
                df_logs['session_id'].nunique(),
                f"{df_logs['processing_time'].mean():.2f}s",
                len(df_logs[df_logs['sources_count'] > 0]),
                f"{df_logs['bot_response_length'].mean():.0f} chars",
                len(df_logs[df_logs['processing_time'] > 5.0])
            ]
        })
        perf_stats.to_excel(writer, sheet_name='Performance Summary', index=False)
    
    async def analyze_for_rag_training(self, logs: List[Dict]) -> Dict[str, Any]:
        """
        Analyze logs to identify areas for RAG improvement.
        
        Returns insights for training data enhancement.
        """
        if not logs:
            return {'error': 'No logs to analyze'}
        
        analysis = {
            'total_conversations': len(logs),
            'unique_sessions': len(set(log['session_id'] for log in logs)),
            'date_range': {
                'earliest': min(log['created_at'] for log in logs),
                'latest': max(log['created_at'] for log in logs)
            }
        }
        
        # Questions without good sources (potential knowledge gaps)
        no_sources = [log for log in logs if log['sources_count'] == 0]
        analysis['knowledge_gaps'] = {
            'count': len(no_sources),
            'percentage': (len(no_sources) / len(logs)) * 100,
            'sample_questions': [log['user_message'][:100] + '...' for log in no_sources[:5]]
        }
        
        # Slow responses (potential optimization needed)
        slow_responses = [log for log in logs if log['processing_time'] > 5.0]
        analysis['performance_issues'] = {
            'slow_responses_count': len(slow_responses),
            'percentage': (len(slow_responses) / len(logs)) * 100,
            'avg_processing_time': sum(log['processing_time'] for log in logs) / len(logs)
        }
        
        # Common question patterns
        question_words = {}
        for log in logs:
            words = log['user_message'].lower().split()
            for word in words:
                if len(word) > 3:  # Filter out short words
                    question_words[word] = question_words.get(word, 0) + 1
        
        # Top 10 most common words in questions
        top_words = sorted(question_words.items(), key=lambda x: x[1], reverse=True)[:10]
        analysis['common_topics'] = dict(top_words)
        
        # Response quality indicators
        short_responses = [log for log in logs if log['bot_response_length'] < 100]
        analysis['response_quality'] = {
            'short_responses_count': len(short_responses),
            'avg_response_length': sum(log['bot_response_length'] for log in logs) / len(logs),
            'responses_with_sources': len([log for log in logs if log['sources_count'] > 0])
        }
        
        return analysis


async def main():
    parser = argparse.ArgumentParser(description='Export and analyze chat logs')
    parser.add_argument('--format', choices=['json', 'csv', 'excel'], default='json',
                       help='Output format')
    parser.add_argument('--output', type=str, help='Output filename')
    parser.add_argument('--days', type=int, help='Number of days to look back')
    parser.add_argument('--session', type=str, help='Specific session ID to export')
    parser.add_argument('--limit', type=int, help='Maximum number of messages')
    parser.add_argument('--analysis', action='store_true', 
                       help='Include analysis sheets (Excel only)')
    parser.add_argument('--rag-analysis', action='store_true',
                       help='Perform RAG training analysis')
    
    args = parser.parse_args()
    
    # Default output filename
    if not args.output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        extension = 'xlsx' if args.format == 'excel' else args.format
        args.output = f'chat_logs_{timestamp}.{extension}'
    
    try:
        exporter = ChatLogExporter()
        
        # Get logs
        logs = await exporter.get_chat_logs(
            days_back=args.days,
            session_id=args.session,
            limit=args.limit
        )
        
        if not logs:
            print("No chat logs found with the specified criteria.")
            return
        
        # Export in requested format
        if args.format == 'json':
            await exporter.export_to_json(logs, args.output)
        elif args.format == 'csv':
            await exporter.export_to_csv(logs, args.output)
        elif args.format == 'excel':
            await exporter.export_to_excel(logs, args.output, args.analysis)
        
        print(f"SUCCESS: Exported {len(logs)} chat messages to {args.output}")
        
        # RAG Analysis
        if args.rag_analysis:
            analysis = await exporter.analyze_for_rag_training(logs)
            analysis_file = args.output.replace('.', '_analysis.')
            if not analysis_file.endswith('.json'):
                analysis_file = analysis_file.rsplit('.', 1)[0] + '_analysis.json'
            
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            print(f"ANALYSIS: RAG Analysis saved to {analysis_file}")
            
            # Print summary
            print("\nRAG Training Insights:")
            print(f"  - Total conversations: {analysis['total_conversations']}")
            print(f"  - Knowledge gaps: {analysis['knowledge_gaps']['count']} ({analysis['knowledge_gaps']['percentage']:.1f}%)")
            print(f"  - Performance issues: {analysis['performance_issues']['slow_responses_count']} slow responses")
            print(f"  - Average processing time: {analysis['performance_issues']['avg_processing_time']:.2f}s")
            print(f"  - Responses with sources: {analysis['response_quality']['responses_with_sources']}")
            
    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        print(f"ERROR: Export failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())