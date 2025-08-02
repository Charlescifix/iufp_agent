#!/usr/bin/env python3
"""
RAG Training Analysis Tool
=========================

Analyze exported chat logs to identify training opportunities for your RAG system.
This script helps you understand:
1. Which questions aren't being answered well (no sources found)
2. Common user query patterns
3. Areas where your knowledge base needs improvement

Usage:
    python analyze_logs_for_training.py chat_logs_file.json
"""

import json
import sys
import re
from collections import Counter, defaultdict
from typing import Dict, List, Any
import argparse


class RAGTrainingAnalyzer:
    def __init__(self, logs_file: str):
        with open(logs_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.messages = self.data.get('messages', [])
        
    def analyze_knowledge_gaps(self) -> Dict[str, Any]:
        """Identify questions that received no source citations (knowledge gaps)."""
        no_sources = [msg for msg in self.messages if msg['sources_count'] == 0]
        
        # Group similar questions
        question_themes = defaultdict(list)
        
        for msg in no_sources:
            question = msg['user_message'].lower()
            
            # Simple thematic categorization
            if any(word in question for word in ['visa', 'student visa', 'immigration']):
                question_themes['Visa & Immigration'].append(msg['user_message'])
            elif any(word in question for word in ['partner school', 'where', 'study']):
                question_themes['Partner Schools & Study Locations'].append(msg['user_message'])
            elif any(word in question for word in ['benefit', 'advantage', 'why']):
                question_themes['IUFP Benefits & Value'].append(msg['user_message'])
            elif any(word in question for word in ['requirement', 'need', 'application']):
                question_themes['Requirements & Applications'].append(msg['user_message'])
            elif any(word in question for word in ['cost', 'fee', 'price', 'payment']):
                question_themes['Costs & Payments'].append(msg['user_message'])
            else:
                question_themes['General/Other'].append(msg['user_message'])
        
        return {
            'total_no_sources': len(no_sources),
            'percentage': (len(no_sources) / len(self.messages)) * 100 if self.messages else 0,
            'themes': dict(question_themes),
            'sample_questions': [msg['user_message'] for msg in no_sources[:10]]
        }
    
    def analyze_query_patterns(self) -> Dict[str, Any]:
        """Analyze common patterns in user queries."""
        all_questions = [msg['user_message'].lower() for msg in self.messages]
        
        # Extract keywords
        all_words = []
        for question in all_questions:
            # Remove punctuation and split
            words = re.findall(r'\b\w+\b', question)
            # Filter out very short words and common stop words
            words = [w for w in words if len(w) > 2 and w not in 
                    {'the', 'and', 'for', 'are', 'can', 'you', 'how', 'what', 'where', 'when', 'why'}]
            all_words.extend(words)
        
        # Most common keywords
        keyword_counts = Counter(all_words)
        
        # Question starters
        question_starters = []
        for question in all_questions:
            first_words = ' '.join(question.split()[:3])
            question_starters.append(first_words)
        
        starter_counts = Counter(question_starters)
        
        return {
            'total_questions': len(all_questions),
            'most_common_keywords': dict(keyword_counts.most_common(15)),
            'common_question_starters': dict(starter_counts.most_common(10)),
            'avg_question_length': sum(len(q.split()) for q in all_questions) / len(all_questions) if all_questions else 0
        }
    
    def suggest_content_additions(self) -> Dict[str, List[str]]:
        """Suggest what content should be added to improve RAG performance."""
        gaps = self.analyze_knowledge_gaps()
        
        suggestions = {}
        
        for theme, questions in gaps['themes'].items():
            if theme == 'Visa & Immigration':
                suggestions[theme] = [
                    "Add comprehensive visa application guides",
                    "Include step-by-step immigration process documents",
                    "Add visa fee schedules and payment information",
                    "Include visa timeline and processing information"
                ]
            elif theme == 'Partner Schools & Study Locations':
                suggestions[theme] = [
                    "Add detailed partner school directory with locations",
                    "Include information about study programs at each location",
                    "Add admission requirements for partner schools",
                    "Include contact information for partner institutions"
                ]
            elif theme == 'IUFP Benefits & Value':
                suggestions[theme] = [
                    "Add comprehensive benefits documentation",
                    "Include success stories and testimonials",
                    "Add career outcomes and university admission statistics",
                    "Include comparison with other foundation programs"
                ]
            elif theme == 'Requirements & Applications':
                suggestions[theme] = [
                    "Add detailed application requirements",
                    "Include step-by-step application process",
                    "Add document checklists and templates",
                    "Include admission criteria and deadlines"
                ]
            elif theme == 'Costs & Payments':
                suggestions[theme] = [
                    "Add comprehensive fee schedules",
                    "Include payment methods and schedules",
                    "Add scholarship and financial aid information",
                    "Include refund policies and procedures"
                ]
        
        return suggestions
    
    def performance_analysis(self) -> Dict[str, Any]:
        """Analyze system performance patterns."""
        processing_times = [msg['processing_time'] for msg in self.messages]
        response_lengths = [msg['bot_response_length'] for msg in self.messages]
        
        # Performance by time of day
        time_performance = defaultdict(list)
        for msg in self.messages:
            created_at = msg['created_at']
            hour = int(created_at.split('T')[1].split(':')[0]) if 'T' in created_at else 12
            time_performance[hour].append(msg['processing_time'])
        
        return {
            'avg_processing_time': sum(processing_times) / len(processing_times) if processing_times else 0,
            'max_processing_time': max(processing_times) if processing_times else 0,
            'min_processing_time': min(processing_times) if processing_times else 0,
            'slow_responses': len([t for t in processing_times if t > 5.0]),
            'avg_response_length': sum(response_lengths) / len(response_lengths) if response_lengths else 0,
            'performance_by_hour': {str(hour): sum(times)/len(times) for hour, times in time_performance.items() if times}
        }
    
    def generate_training_report(self) -> str:
        """Generate a comprehensive training improvement report."""
        gaps = self.analyze_knowledge_gaps()
        patterns = self.analyze_query_patterns()
        suggestions = self.suggest_content_additions()
        performance = self.performance_analysis()
        
        report = f"""
# RAG Training Improvement Report
## Generated from {len(self.messages)} chat conversations

## 🚨 CRITICAL FINDINGS
- **{gaps['percentage']:.1f}%** of questions received NO source citations
- This indicates major knowledge gaps in your RAG system
- **{performance['slow_responses']}** responses took over 5 seconds

## 📊 KNOWLEDGE GAPS ANALYSIS
Total questions without sources: {gaps['total_no_sources']}

### By Theme:
"""
        
        for theme, questions in gaps['themes'].items():
            report += f"\n**{theme}**: {len(questions)} questions\n"
            if questions:
                report += f"  Sample: \"{questions[0][:80]}...\"\n"
        
        report += f"""
## 🔍 USER QUERY PATTERNS
- Average question length: {patterns['avg_question_length']:.1f} words
- Most discussed topics: {', '.join(list(patterns['most_common_keywords'].keys())[:5])}

### Top Keywords:
"""
        for keyword, count in list(patterns['most_common_keywords'].items())[:10]:
            report += f"- **{keyword}**: {count} mentions\n"
        
        report += "\n## 💡 RECOMMENDED ACTIONS\n"
        
        priority_order = ['Visa & Immigration', 'Partner Schools & Study Locations', 'IUFP Benefits & Value', 
                         'Requirements & Applications', 'Costs & Payments']
        
        for i, theme in enumerate(priority_order, 1):
            if theme in suggestions:
                report += f"\n### Priority {i}: {theme}\n"
                for suggestion in suggestions[theme]:
                    report += f"- {suggestion}\n"
        
        report += f"""
## ⚡ PERFORMANCE INSIGHTS
- Average response time: {performance['avg_processing_time']:.2f}s
- Longest response: {performance['max_processing_time']:.2f}s
- Responses with sources: {len([msg for msg in self.messages if msg['sources_count'] > 0])}
- Average response length: {performance['avg_response_length']:.0f} characters

## 📋 IMMEDIATE ACTION ITEMS
1. **Add missing content** for the top 3 question themes
2. **Optimize search performance** to reduce response times
3. **Review document chunking** to improve source matching
4. **Test retrieval settings** (vector/BM25 weights) with sample questions

## 📝 SAMPLE UNANSWERED QUESTIONS TO TEST WITH:
"""
        
        for i, question in enumerate(gaps['sample_questions'][:5], 1):
            report += f"{i}. \"{question}\"\n"
        
        return report


def main():
    parser = argparse.ArgumentParser(description='Analyze chat logs for RAG training improvements')
    parser.add_argument('logs_file', help='Path to exported chat logs JSON file')
    parser.add_argument('--output', help='Output report file (default: prints to console)')
    
    args = parser.parse_args()
    
    try:
        analyzer = RAGTrainingAnalyzer(args.logs_file)
        report = analyzer.generate_training_report()
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Training report saved to {args.output}")
        else:
            print(report)
            
    except FileNotFoundError:
        print(f"ERROR: Could not find logs file: {args.logs_file}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Analysis failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()