# guardrails.py

import os
import re


class GuardrailsManager:
    def __init__(self):
        self.input_validators = [
            self.validate_input_length,
            self.check_injection_patterns,
            self.validate_content_safety,
            self.check_document_context
        ]
        self.output_validators = [
            self.validate_response_accuracy,
            self.check_hallucination,
            self.verify_source_grounding
        ]

    # Input Validators

    def validate_input_length(self, query: str, *args) -> bool:
        """Prevent buffer overflow attacks by restricting input length."""
        if len(query) > 2000:
            raise ValueError("Query too long")
        return True

    def check_injection_patterns(self, query: str, *args) -> str:
        """Detect and sanitize potential injection attempts in user input."""
        dangerous_patterns = [
            r'<script.*?>.*?</script>',  # XSS
            r'(union|select|drop|delete|insert|update)\s+',  # SQL injection
            r'(eval|exec|system|shell_exec)\s*\(',  # Code injection
        ]

        cleaned_query = query
        for pattern in dangerous_patterns:
            cleaned_query = re.sub(pattern, '', cleaned_query, flags=re.IGNORECASE)

        return cleaned_query

    def validate_content_safety(self, query: str, *args) -> bool:
        """Check for harmful or inappropriate keywords in the input."""
        harmful_keywords = [
            'password', 'secret', 'private key', 'confidential',
            'hack', 'exploit', 'malware', 'virus'
        ]

        query_lower = query.lower()
        for keyword in harmful_keywords:
            if keyword in query_lower:
                raise ValueError(f"Query contains sensitive keyword: {keyword}")

        return True

    def check_document_context(self, query: str, available_docs: list) -> bool:
        """Ensure relevant document context is available for the query."""
        if not available_docs:
            raise ValueError("No documents available for querying")
        return True

    # Output Validators

    def validate_response_accuracy(self, response: str, sources: list) -> bool:
        """Verify that the AI response is grounded in source documents."""
        if not sources:
            return False

        source_keywords = set()
        for source in sources:
            source_keywords.update(source.split())

        response_words = set(response.lower().split())
        overlap = len(source_keywords.intersection(response_words))

        # Require at least 10% overlap to consider grounded
        return overlap / len(response_words) >= 0.1 if len(response_words) > 0 else False

    def check_hallucination(self, response: str, sources: list) -> bool:
        """Detect if response contains hallucinated or fabricated content.

        Placeholder: Implement more sophisticated hallucination detection if desired.
        """
        # Simple placeholder check - could be improved
        return self.validate_response_accuracy(response, sources)

    def verify_source_grounding(self, response: str, sources: list) -> bool:
        """Additional factual consistency or grounding checks if needed."""
        # Placeholder identical to validate_response_accuracy for now
        return self.validate_response_accuracy(response, sources)

    # Metadata Reference Validators for Images/Tables

    def validate_metadata_reference(self, ref: str) -> str:
        """
        Sanitize metadata references (e.g., image/table file names).
        Prevent path traversal, unusual characters, or malicious payloads.
        """
        if not re.match(r'^[\w\-. ]+$', ref):
            raise ValueError("Invalid reference detected in metadata")

        normalized_ref = os.path.normpath(ref)
        if normalized_ref.startswith("..") or os.path.isabs(normalized_ref):
            raise ValueError("Path traversal detected in metadata reference")

        return normalized_ref

    def validate_output_references(self, response_references: list, valid_refs: list) -> bool:
        """
        Check that all image/table references mentioned in LLM output are known and valid.
        """
        for ref in response_references:
            try:
                sanitized_ref = self.validate_metadata_reference(ref)
            except ValueError:
                return False
            if sanitized_ref not in valid_refs:
                return False
        return True

    def extract_references_from_response(self, response: str) -> list:
        """
        Extract image/table references from LLM response text.
        Supports patterns like: [Image: filename.png], [Table: table_1]
        """
        pattern = r'\[Image:\s*([^\]]+)\]|\[Table:\s*([^\]]+)\]'
        matches = re.findall(pattern, response)
        refs = []
        for img_ref, table_ref in matches:
            if img_ref:
                refs.append(img_ref.strip())
            if table_ref:
                refs.append(table_ref.strip())
        return refs

    # Combined Validator Methods

    def validate_input(self, query: str, context: dict = None) -> str:
        """
        Run all input validators sequentially on the user query.
        Optionally uses context to access available documents.
        """
        for validator in self.input_validators:
            if validator == self.check_injection_patterns:
                query = validator(query)
            else:
                validator(query, context.get('documents', []) if context else [])
        return query

    def validate_output(self, response: str, sources: list, references: list = None) -> bool:
        """
        Run all output validators sequentially.
        If references to images/tables exist, check their validity.
        """
        for validator in self.output_validators:
            print("vo: =>   ", validator(response, sources))
            if not validator(response, sources):
                return False

        if references:
            response_refs = self.extract_references_from_response(response)
            if not self.validate_output_references(response_refs, references):
                return False

        return True
