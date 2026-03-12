import uuid
import time
import json
import traceback
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
import io

from ingestion.loader import ingest
from handlers.text_handler import handle_text
from handlers.binary_handler import handle_binary
from handlers.mixed_handler import handle_mixed

@dataclass
class ProcessingResult:
    file_name: str
    document_id: str
    status: str  # 'success' or 'failed'
    processing_time: float
    detected_format: Optional[str] = None
    mime_type: Optional[str] = None
    routing_target: Optional[str] = None
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

@dataclass
class BatchJob:
    job_id: str
    files: List[Dict[str, Any]]
    status: str  # 'pending', 'processing', 'completed', 'failed'
    created_at: float
    completed_at: Optional[float] = None
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    results: List[Dict[str, Any]] = None
    errors: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.results is None: self.results = []
        if self.errors is None: self.errors = []
        self.total_files = len(self.files)

class WrapperFile:
    """Mock file object for compatibility with ingest() function"""
    def __init__(self, name, content):
        self.name = name
        self.content = content
    def read(self):
        return self.content

def _process_single_file_task(file_data: Dict[str, Any]) -> Dict[str, Any]:
    """Standalone function for worker processes to avoid pickling 'self'"""
    start_time = time.time()
    file_name = file_data.get('name', 'unknown')
    file_bytes = file_data.get('bytes')
    
    try:
        # Create a mock file object for ingest()
        file_obj = WrapperFile(file_name, file_bytes)
        doc = ingest(file=file_obj)
        
        # Route and handle based on the routing target
        output = None
        if doc.routing_target == "text_handler":
            output = handle_text(doc)
        elif doc.routing_target == "binary_handler":
            output = handle_binary(doc)
        elif doc.routing_target == "mixed_handler":
            output = handle_mixed(doc)
        else:
            # Explicit error for unhandled routing targets
            raise ValueError(f"Unhandled routing target: {doc.routing_target}. "
                           f"File: {file_name}, Format: {doc.detected_format}, "
                           f"MIME: {doc.mime_type}")
        
        res = ProcessingResult(
            file_name=file_name,
            document_id=doc.document_id,
            status='success',
            processing_time=time.time() - start_time,
            detected_format=doc.detected_format,
            mime_type=doc.mime_type,
            routing_target=doc.routing_target,
            output_data=asdict(output) if output else None
        )
        return asdict(res)
        
    except Exception as e:
        res = ProcessingResult(
            file_name=file_name,
            document_id="N/A",
            status='failed',
            processing_time=time.time() - start_time,
            error_message=str(e)
        )
        return asdict(res)

class BatchProcessor:
    def __init__(self, max_workers: int = 4, use_processes: bool = True, 
                 save_outputs: bool = True, output_dir: str = "outputs/batch"):
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.save_outputs = save_outputs
        self.output_dir = Path(output_dir)
        self.jobs: Dict[str, BatchJob] = {}
        
        if self.save_outputs:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_batch_job(self, files: List[Dict[str, Any]]) -> str:
        """Create a new batch job and return its ID"""
        job_id = str(uuid.uuid4())
        job = BatchJob(
            job_id=job_id,
            files=files,
            status='pending',
            created_at=time.time()
        )
        self.jobs[job_id] = job
        return job_id

    def process_batch(self, job_id: str) -> BatchJob:
        """Process all files in a batch job"""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self.jobs[job_id]
        job.status = 'processing'
        
        Executor = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        with Executor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(_process_single_file_task, f): f 
                for f in job.files
            }
            
            for future in as_completed(future_to_file):
                try:
                    res_dict = future.result()
                    if res_dict['status'] == 'success':
                        job.results.append(res_dict)
                        job.processed_files += 1
                    else:
                        job.errors.append({
                            'file_name': res_dict['file_name'],
                            'error_message': res_dict['error_message'],
                            'status': 'failed',
                            'processing_time': res_dict['processing_time']
                        })
                        job.failed_files += 1
                except Exception as exc:
                    file_data = future_to_file[future]
                    fname = file_data.get('name', 'unknown')
                    job.errors.append({
                        'file_name': fname,
                        'error_message': f"Internal error: {str(exc)}",
                        'status': 'failed',
                        'processing_time': 0
                    })
                    job.failed_files += 1
                
        job.status = 'completed'
        job.completed_at = time.time()
        
        if self.save_outputs:
            self._save_job_results(job)
            
        return job

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get the current status of a batch job"""
        if job_id not in self.jobs:
            return {"error": "Job not found"}
        job = self.jobs[job_id]
        return {
            "job_id": job.job_id,
            "status": job.status,
            "total_files": job.total_files,
            "processed_files": job.processed_files,
            "failed_files": job.failed_files,
            "progress": (job.processed_files + job.failed_files) / job.total_files * 100 if job.total_files > 0 else 0
        }

    def _process_single_file(self, file_data: Dict[str, Any]) -> ProcessingResult:
        """Process a single file (required by specification)"""
        res_dict = _process_single_file_task(file_data)
        return ProcessingResult(**res_dict)

    def _save_job_results(self, job: BatchJob):
        """Save batch job results to JSON file"""
        output_path = self.output_dir / f"batch_{job.job_id}.json"
        data = {
            "job_id": job.job_id,
            "summary": {
                "total_files": job.total_files,
                "processed_files": job.processed_files,
                "failed_files": job.failed_files,
                "processing_time": job.completed_at - job.created_at if job.completed_at else 0
            },
            "results": job.results,
            "errors": job.errors
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

def process_files_batch(files: List[Dict[str, Any]], max_workers: int = 4, 
                        save_outputs: bool = True) -> Dict[str, Any]:
    """Simple function interface for batch processing"""
    processor = BatchProcessor(max_workers=max_workers, save_outputs=save_outputs)
    job_id = processor.create_batch_job(files)
    job = processor.process_batch(job_id)
    
    return {
        "job_id": job.job_id,
        "summary": {
            "total_files": job.total_files,
            "processed_files": job.processed_files,
            "failed_files": job.failed_files,
            "processing_time": job.completed_at - job.created_at
        },
        "results": job.results,
        "errors": job.errors
    }