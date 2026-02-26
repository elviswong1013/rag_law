from typing import Dict, Any, List
from llm import OpenAILLM
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    relevance_score: float
    accuracy_score: float
    completeness_score: float
    clarity_score: float
    overall_score: float
    feedback: str
    details: Dict[str, Any]


class LLMJudge:
    def __init__(self, llm: OpenAILLM, metrics: List[str] = None):
        self.llm = llm
        self.metrics = metrics or ["relevance", "accuracy", "completeness", "clarity"]
    
    def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Dict[str, Any]],
        reference_answer: str = None
    ) -> EvaluationResult:
        context = self._build_context(retrieved_docs)
        
        evaluation_prompt = self._build_evaluation_prompt(
            query=query,
            answer=answer,
            context=context,
            reference_answer=reference_answer,
            metrics=self.metrics
        )
        
        response = self.llm.generate(
            evaluation_prompt,
            temperature=0.3,
            max_tokens=1500
        )
        
        return self._parse_evaluation(response)
    
    def evaluate_batch(
        self,
        queries: List[str],
        answers: List[str],
        retrieved_docs_list: List[List[Dict[str, Any]]],
        reference_answers: List[str] = None
    ) -> List[EvaluationResult]:
        results = []
        
        for i, query in enumerate(queries):
            ref_answer = reference_answers[i] if reference_answers else None
            result = self.evaluate(
                query=query,
                answer=answers[i],
                retrieved_docs=retrieved_docs_list[i],
                reference_answer=ref_answer
            )
            results.append(result)
        
        return results
    
    def _build_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        if not retrieved_docs:
            return "No documents were retrieved."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"[Document {i}]")
            context_parts.append(doc.get('content', ''))
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _build_evaluation_prompt(
        self,
        query: str,
        answer: str,
        context: str,
        reference_answer: str = None,
        metrics: List[str] = None
    ) -> str:
        metrics = metrics or self.metrics
        
        prompt = f"""你是一个专业的法律问答评估专家。请根据以下信息评估回答的质量。

问题: {query}

检索到的文档:
{context}

生成的回答:
{answer}
"""
        
        if reference_answer:
            prompt += f"""
参考答案:
{reference_answer}
"""
        
        prompt += f"""
请从以下维度评估回答的质量（每个维度评分范围为0-10分）:

1. 相关性: 回答是否直接回答了问题
2. 准确性: 回答是否基于检索到的文档且信息准确
3. 完整性: 回答是否涵盖了问题的所有方面
4. 清晰性: 回答是否清晰易懂，逻辑是否连贯

请以JSON格式返回评估结果，格式如下:
{{
    "relevance_score": <0-10>,
    "accuracy_score": <0-10>,
    "completeness_score": <0-10>,
    "clarity_score": <0-10>,
    "overall_score": <0-10>,
    "feedback": "<详细的反馈意见>",
    "details": {{
        "strengths": ["优点1", "优点2"],
        "weaknesses": ["缺点1", "缺点2"],
        "suggestions": ["改进建议1", "改进建议2"]
    }}
}}

只返回JSON，不要包含其他内容。"""
        
        return prompt
    
    def _parse_evaluation(self, response: str) -> EvaluationResult:
        import json
        import re
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                eval_dict = json.loads(json_match.group())
                
                return EvaluationResult(
                    relevance_score=eval_dict.get('relevance_score', 0.0),
                    accuracy_score=eval_dict.get('accuracy_score', 0.0),
                    completeness_score=eval_dict.get('completeness_score', 0.0),
                    clarity_score=eval_dict.get('clarity_score', 0.0),
                    overall_score=eval_dict.get('overall_score', 0.0),
                    feedback=eval_dict.get('feedback', ''),
                    details=eval_dict.get('details', {})
                )
        except Exception as e:
            print(f"Error parsing evaluation: {e}")
        
        return EvaluationResult(
            relevance_score=0.0,
            accuracy_score=0.0,
            completeness_score=0.0,
            clarity_score=0.0,
            overall_score=0.0,
            feedback="Failed to parse evaluation",
            details={}
        )
    
    def compare_answers(
        self,
        query: str,
        answer1: str,
        answer2: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        context = self._build_context(retrieved_docs)
        
        comparison_prompt = f"""请比较以下两个回答的质量。

问题: {query}

检索到的文档:
{context}

回答1:
{answer1}

回答2:
{answer2}

请从以下维度比较两个回答:
1. 相关性
2. 准确性
3. 完整性
4. 清晰性

请以JSON格式返回比较结果:
{{
    "winner": "answer1" 或 "answer2" 或 "tie",
    "reasoning": "<比较理由>",
    "answer1_scores": {{
        "relevance": <0-10>,
        "accuracy": <0-10>,
        "completeness": <0-10>,
        "clarity": <0-10>
    }},
    "answer2_scores": {{
        "relevance": <0-10>,
        "accuracy": <0-10>,
        "completeness": <0-10>,
        "clarity": <0-10>
    }}
}}

只返回JSON，不要包含其他内容。"""
        
        response = self.llm.generate(
            comparison_prompt,
            temperature=0.3,
            max_tokens=1000
        )
        
        try:
            import json
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"Error parsing comparison: {e}")
        
        return {
            "winner": "tie",
            "reasoning": "Failed to parse comparison",
            "answer1_scores": {},
            "answer2_scores": {}
        }
    
    def generate_improvement_suggestions(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> List[str]:
        context = self._build_context(retrieved_docs)
        
        improvement_prompt = f"""请为以下回答提供改进建议。

问题: {query}

检索到的文档:
{context}

当前回答:
{answer}

请提供3-5条具体的改进建议，每条建议应该清晰可执行。

以JSON格式返回:
{{
    "suggestions": [
        "建议1",
        "建议2",
        "建议3"
    ]
}}

只返回JSON，不要包含其他内容。"""
        
        response = self.llm.generate(
            improvement_prompt,
            temperature=0.5,
            max_tokens=800
        )
        
        try:
            import json
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result.get('suggestions', [])
        except Exception as e:
            print(f"Error parsing suggestions: {e}")
        
        return []
