import json
import tempfile
from logging import Logger

from openai import OpenAI
from progress.bar import Bar

from aius.analyze.backend import Backend
from aius.analyze.data_models import Document, ModelResponse

MAX_BATCH_FILE_BYTES: int = 10 * 1024 * 1024


class OpenAIBatchBackend(Backend):
    def __init__(
        self,
        logger: Logger,
        auth_key: str,
        model_name: str = "gpt-5.4-nano-2026-03-17",
        **kwargs,
    ) -> None:
        super().__init__(
            name="openaibatch",
            logger=logger,
            model_name=model_name,
        )

        self.openai_client = OpenAI(
            api_key=auth_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def _build_batch_input_files(
        self,
        documents: list[Document],
        system_prompt: str,
    ) -> tuple[list[str], dict[str, str]]:
        filenames: list[str] = []
        doc_map: dict[str, str] = {}

        current_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".jsonl",
            delete=False,
            encoding="utf-8",
        )
        current_size: int = 0

        for index, document in enumerate(documents):
            custom_id = f"doc-{index}"
            doc_map[custom_id] = document.doi

            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": self.model_name,
                    "input": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": document.content},
                    ],
                    "max_output_tokens": 20000,
                    "reasoning": {"effort": "high", "summary": "detailed"},
                    "text": {"format": {"type": "json_object"}},
                },
            }

            line: str = json.dumps(request, ensure_ascii=False) + "\n"
            line_size: int = len(line.encode("utf-8"))

            if line_size > MAX_BATCH_FILE_BYTES:
                raise ValueError(
                    f"Single request exceeds batch size limit ({line_size} bytes): {document.doi}"
                )

            if current_size + line_size > MAX_BATCH_FILE_BYTES:
                current_file.close()
                filenames.append(current_file.name)
                current_file = tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".jsonl",
                    delete=False,
                    encoding="utf-8",
                )
                current_size = 0

            current_file.write(line)
            current_size += line_size

        current_file.close()
        filenames.append(current_file.name)

        with open("doc_map.txt", "w", encoding="utf-8") as fp:
            json.dump(obj=doc_map, fp=fp, indent=4)

        return filenames, doc_map

    def inference_document(
        self, document: Document, system_prompt: str
    ) -> ModelResponse:
        return ModelResponse(
            doi=document.doi,
            system_prompt=system_prompt,
            user_prompt=document.content,
            model_reasoning="",
            model_response="",
            compute_time_seconds=0,
        )

    def inference_documents(
        self,
        documents: list[Document],
        system_prompt: str,
    ) -> list[ModelResponse]:
        self.logger.info(
            "Building OpenAI batch input for %d documents...",
            len(documents),
        )

        input_filenames, _ = self._build_batch_input_files(
            documents=documents,
            system_prompt=system_prompt,
        )

        self.logger.info("Wrote %d batch shard(s)", len(input_filenames))

        with Bar("Uploading batch input files...", max=len(input_filenames)) as bar:
            fn: str
            for fn in input_filenames:
                self.logger.info("Batch shard: %s", fn)
                with open(fn, "rb") as f:
                    self.openai_client.files.create(
                        file=f,
                        purpose="batch",
                    )
                bar.next()
                continue

            # # Create batch job
            # self.logger.info("Creating batch job...")

            # batch = self.openai_client.batches.create(
            #     input_file_id=input_file.id,
            #     endpoint="/v1/responses",
            #     completion_window="24h",
            # )

            # self.logger.info(
            #     "Batch created: %s",
            #     batch.id,
            # )

            # bar.next()

        return [
            ModelResponse(
                doi=document.doi,
                system_prompt=system_prompt,
                user_prompt=document.content,
                model_reasoning="",
                model_response="",
                compute_time_seconds=0,
            )
            for document in documents
        ]
