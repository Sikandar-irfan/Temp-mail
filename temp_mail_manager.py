import os
import re
import uuid
import json
import queue
import string
import random
import logging
import requests
import threading
from time import time, sleep
from typing import Dict, List, Optional
from string import Template
from datetime import datetime
from hashlib import md5
from threading import Lock
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    import aiosmtplib
except ImportError:
    aiosmtplib = None

try:
    from faker import Faker
except ImportError:
    Faker = None

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from mimetypes import guess_type
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Setup logging
def setup_logging():
    """Configure logging to write to files instead of console"""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup file handlers
    app_log = os.path.join(log_dir, 'app.log')
    error_log = os.path.join(log_dir, 'error.log')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    root_logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # Setup file handlers
    file_handler = RotatingFileHandler(
        app_log,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    error_handler = RotatingFileHandler(
        error_log,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    
    # Prevent logs from being printed to console
    root_logger.propagate = False

setup_logging()

fake = Faker()

@dataclass
class EmailMessage:
    """Email message data class"""
    id: str
    sender: str
    subject: str
    body: str
    date: str
    attachments: List[Dict] = None

class RateLimiter:
    """Rate limiter using token bucket algorithm with request tracking"""
    def __init__(self, tokens_per_second: float = 2.0, burst_size: int = 10, 
                 max_requests: int = 100, time_window: int = 3600):
        self.tokens_per_second = tokens_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time()
        self.lock = Lock()
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []

    def acquire(self, tokens: int = 1) -> bool:
        with self.lock:
            now = time()
            # Clean up old requests
            self.requests = [req_time for req_time in self.requests 
                           if now - req_time < self.time_window]
            
            # Check request limit
            if len(self.requests) >= self.max_requests:
                return False
                
            # Update tokens
            time_passed = now - self.last_update
            self.tokens = min(
                self.burst_size,
                self.tokens + time_passed * self.tokens_per_second
            )
            self.last_update = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                self.requests.append(now)
                return True
            return False

    def get_remaining_requests(self) -> int:
        """Get remaining requests in the current time window"""
        with self.lock:
            now = time()
            self.requests = [req_time for req_time in self.requests 
                           if now - req_time < self.time_window]
            return self.max_requests - len(self.requests)

def rate_limit(tokens: int = 1):
    """Decorator for rate limiting"""
    def decorator(func):
        if not hasattr(func, '_rate_limiter'):
            func._rate_limiter = RateLimiter(2.0, 10)  # 2 requests per second, burst of 10

        @wraps(func)
        def wrapper(*args, **kwargs):
            while not func._rate_limiter.acquire(tokens):
                sleep(0.1)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def retry_with_backoff(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504)):
    """Create a retry session with exponential backoff"""
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

class APIError(Exception):
    """Base exception for API errors with detailed information"""
    def __init__(self, message: str, status_code: Optional[int] = None, 
                 response_body: Optional[str] = None, request_info: Optional[Dict] = None):
        self.status_code = status_code
        self.response_body = response_body
        self.request_info = request_info
        super().__init__(self._format_message(message))

    def _format_message(self, message: str) -> str:
        parts = [f"API Error: {message}"]
        if self.status_code:
            parts.append(f"Status Code: {self.status_code}")
        if self.response_body:
            parts.append(f"Response: {self.response_body}")
        if self.request_info:
            parts.append(f"Request: {json.dumps(self.request_info, indent=2)}")
        return "\n".join(parts)

@dataclass
class ProviderConfig:
    """Base configuration for email providers"""
    retry_attempts: int = 3
    timeout_seconds: int = 10
    max_attachment_size: int = 10 * 1024 * 1024  # 10MB

@dataclass
class TenMinuteMailConfig(ProviderConfig):
    """Configuration for TenMinuteMailAPI"""
    email_duration_minutes: int = 10
    auto_extend: bool = True

@dataclass
class DisposableMailConfig(ProviderConfig):
    """Configuration for DisposableMailAPI"""
    preferred_domain: Optional[str] = None
    email_prefix: Optional[str] = None

@dataclass
class MohmalConfig(ProviderConfig):
    """Configuration for MohmalAPI"""
    language: str = 'en'
    email_lifetime_hours: int = 24

class EmailProviderError(Exception):
    """Base exception for email provider errors"""
    pass

class EmailGenerationError(EmailProviderError):
    """Raised when email generation fails"""
    pass

class MessageFetchError(EmailProviderError):
    """Raised when fetching messages fails"""
    pass

class AuthenticationError(EmailProviderError):
    """Raised when authentication fails"""
    pass

class EmailTemplate:
    def __init__(self, name: str, subject_template: str, body_template: str):
        self.name = name
        self.subject_template = Template(subject_template)
        self.body_template = Template(body_template)

    def render(self, **kwargs) -> Dict[str, str]:
        return {
            'subject': self.subject_template.render(**kwargs),
            'body': self.body_template.render(**kwargs)
        }

class EmailMessage:
    def __init__(self, data: Dict):
        self.subject = data.get('subject', '')
        self.sender = data.get('from', '')
        self.body = data.get('body', '')
        self.received_at = data.get('received_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.attachments = data.get('attachments', [])
        self.message_id = md5(f"{self.subject}{self.sender}{self.received_at}".encode()).hexdigest()
        self.categories = []
        self.is_otp = self._check_if_otp()
        self.extracted_otp = self._extract_otp() if self.is_otp else None
        self.sentiment = self._analyze_sentiment()
        self.priority = self._calculate_priority()
        self.html_content = data.get('html', '')
        self.links = self._extract_links()
        self.spam_score = self._calculate_spam_score()

    def _check_if_otp(self) -> bool:
        otp_keywords = ['otp', 'verification', 'verify', 'code', 'confirmation']
        return any(keyword in self.subject.lower() or keyword in self.body.lower() 
                  for keyword in otp_keywords)

    def _extract_otp(self) -> Optional[str]:
        patterns = [
            r'\b\d{4,8}\b',
            r'[A-Z0-9]{6,8}',
            r'(?i)code[:\s]+([A-Z0-9]{4,8})',
            r'(?i)otp[:\s]+([A-Z0-9]{4,8})'
        ]
        
        text = f"{self.subject} {self.body}"
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0]
        return None

    def _analyze_sentiment(self) -> str:
        # Basic sentiment analysis based on keywords
        positive_words = {'success', 'approved', 'confirmed', 'welcome', 'thank'}
        negative_words = {'fail', 'reject', 'error', 'invalid', 'expired'}
        
        text = f"{self.subject} {self.body}".lower()
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        return 'neutral'

    def _calculate_priority(self) -> int:
        priority = 0
        if self.is_otp:
            priority += 3
        if 'urgent' in self.subject.lower():
            priority += 2
        if self.sentiment == 'negative':
            priority += 1
        return min(priority, 5)

    def _extract_links(self) -> List[str]:
        if self.html_content:
            soup = BeautifulSoup(self.html_content, 'html.parser')
            return [a.get('href') for a in soup.find_all('a', href=True)]
        return []

    def _calculate_spam_score(self) -> float:
        score = 0.0
        text = f"{self.subject} {self.body}".lower()
        
        # Common spam indicators
        spam_indicators = {
            r'\$\d+': 0.3,  # Money amounts
            r'(?i)urgent': 0.2,
            r'(?i)winner': 0.4,
            r'(?i)lottery': 0.4,
            r'(?i)password': 0.2,
            r'(?i)account.*suspend': 0.3,
        }
        
        for pattern, weight in spam_indicators.items():
            if re.search(pattern, text):
                score += weight
        
        # Check for excessive capitalization
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        if caps_ratio > 0.3:
            score += 0.3
        
        return min(score, 1.0)

    def to_dict(self) -> Dict:
        return {
            'subject': self.subject,
            'sender': self.sender,
            'body': self.body,
            'received_at': self.received_at,
            'attachments': self.attachments,
            'message_id': self.message_id,
            'categories': self.categories,
            'is_otp': self.is_otp,
            'extracted_otp': self.extracted_otp,
            'sentiment': self.sentiment,
            'priority': self.priority,
            'links': self.links,
            'spam_score': self.spam_score
        }

class EmailForwarder:
    def __init__(self):
        self.smtp_settings = {
            'hostname': 'smtp.gmail.com',
            'port': 587,
            'use_tls': True
        }
        if aiosmtplib is None:
            logging.warning("aiosmtplib not installed. Email forwarding will be disabled.")

    async def send_email(self, 
                        from_addr: str,
                        to_addr: str,
                        subject: str,
                        body: str,
                        html_content: Optional[str] = None,
                        attachments: Optional[List[Dict]] = None) -> bool:
        """
        Send an email using SMTP
        
        Args:
            from_addr: Sender email address
            to_addr: Recipient email address
            subject: Email subject
            body: Plain text body
            html_content: Optional HTML content
            attachments: List of attachment dictionaries with 'content' and 'filename' keys
        """
        if aiosmtplib is None:
            logging.error("Cannot send email. aiosmtplib is not installed.")
            return False
        
        try:
            message = MIMEMultipart('alternative')
            message['From'] = from_addr
            message['To'] = to_addr
            message['Subject'] = subject

            # Add plain text body
            message.attach(MIMEText(body, 'plain'))

            # Add HTML content if provided
            if html_content:
                message.attach(MIMEText(html_content, 'html'))

            # Add attachments if any
            if attachments:
                for attachment in attachments:
                    filename = attachment.get('filename')
                    content = attachment.get('content')
                    
                    if filename and content:
                        # Determine content type
                        content_type, _ = guess_type(filename)
                        if content_type is None:
                            content_type = 'application/octet-stream'
                        
                        main_type, sub_type = content_type.split('/', 1)
                        
                        # Decode base64 content if needed
                        if isinstance(content, str):
                            try:
                                content = b64decode(content)
                            except:
                                content = content.encode()
                        
                        # Create attachment part
                        att = MIMEApplication(content, _subtype=sub_type)
                        att.add_header('Content-Disposition', 'attachment', filename=filename)
                        message.attach(att)

            # Connect and send
            smtp = SMTP(
                hostname=self.smtp_settings['hostname'],
                port=self.smtp_settings['port'],
                use_tls=self.smtp_settings['use_tls']
            )

            await smtp.connect()
            await smtp.send_message(message)
            await smtp.quit()

            return True

        except Exception as e:
            logging.error(f"Error sending email: {str(e)}")
            return False

    def forward_email(self, from_email: str, to_email: str, message_id: str) -> bool:
        """Synchronous wrapper for send_email"""
        return asyncio.run(self.send_email(
            from_addr=from_email,
            to_addr=to_email,
            subject="Fwd: " + message_id,
            body="Forwarded message",
            html_content=None,
            attachments=None
        ))

class TempMailAPI(ABC):
    """Abstract base class for temporary email providers"""
    
    @abstractmethod
    def generate_email(self) -> Optional[str]:
        """Generate a new email address"""
        pass
        
    @abstractmethod
    def get_messages(self, email: str) -> List[Dict]:
        """Get messages for an email address"""
        pass
        
    @abstractmethod
    def get_message(self, email: str, message_id: str) -> Optional[Dict]:
        """Get a specific message"""
        pass
        
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get provider name"""
        pass
        
    @abstractmethod
    def get_available_domains(self) -> List[str]:
        """Get list of available domains"""
        pass

class TempMailManager:
    def __init__(self):
        """Initialize TempMailManager"""
        # Setup logging
        self.logger = logging.getLogger('temp_mail_manager')
        
        # Setup data directory and file
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        self.data_file = os.path.join(self.data_dir, 'emails.json')
        
        # Initialize data
        self.emails = {}
        self.providers = {}
        self.load_data()

    def add_email(self, email: str, provider: TempMailAPI) -> bool:
        """Add a new email"""
        try:
            if email not in self.emails:
                provider_class = provider.__class__.__name__
                self.emails[email] = {
                    'provider_class': provider_class,
                    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                self.logger.info(f"Added email {email} with provider {provider_class}")
                self.save_data()
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error adding email: {str(e)}")
            return False

    def get_provider_for_email(self, email: str) -> Optional[TempMailAPI]:
        """Get provider instance for an email"""
        try:
            if email in self.emails:
                provider_class = self.emails[email].get('provider_class')
                self.logger.info(f"Getting provider {provider_class} for email {email}")
                
                if provider_class == 'GuerrillaMailAPI':
                    return GuerrillaMailAPI()
                elif provider_class == 'DisposableMailAPI':
                    return DisposableMailAPI()
                elif provider_class == 'YopMailAPI':
                    return YopMailAPI()
                elif provider_class == 'TempMailOrgAPI':
                    return TempMailOrgAPI()
        
            # Try domain-based lookup as fallback
            domain = email.split('@')[1]
            provider = self.get_provider_by_domain(domain)
            if provider:
                # Update stored provider
                self.emails[email] = {
                    'provider_class': provider.__class__.__name__,
                    'created_at': self.emails.get(email, {}).get('created_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                }
                self.save_data()
                return provider
            
            return None
        except Exception as e:
            self.logger.error(f"Error getting provider for email: {str(e)}")
            return None

    def get_active_emails(self) -> List[str]:
        """
        Get list of active emails, cleaning up old ones
        
        Returns:
            List[str]: List of active email addresses
        """
        try:
            current_time = datetime.now()
            active_emails = []
            
            for email, data in self.emails.items():
                # Skip if no creation time
                if 'created_at' not in data:
                    continue
                    
                # Parse creation time
                created_at = datetime.fromisoformat(data['created_at'])
                
                # Check if email is within active window (24 hours)
                if (current_time - created_at).total_seconds() < 24 * 3600:
                    active_emails.append(email)
                else:
                    # Clean up old email
                    self.delete_email(email)
                    
            return active_emails
            
        except Exception as e:
            self.logger.error(f"Error getting active emails: {str(e)}")
            return []

    def delete_email(self, email: str) -> bool:
        """Delete an email"""
        try:
            if email in self.emails:
                del self.emails[email]
                self.save_data()
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error deleting email: {str(e)}")
            return False

    def load_data(self):
        """Load saved email data"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    self.emails = json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            self.emails = {}

    def save_data(self):
        """Save email data"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.emails, f, indent=4)
            self.logger.info("Email data saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")

    def get_messages(self, email: str) -> List[Dict]:
        """
        Get messages for an email address
        
        Args:
            email (str): Email address to get messages for
            
        Returns:
            List[Dict]: List of message dictionaries
        """
        try:
            provider = self.get_provider_for_email(email)
            if not provider:
                raise EmailProviderError(f"No provider found for {email}")

            messages = provider.get_messages(email)
            if not messages:
                return []

            # Ensure all message fields are strings for display
            formatted_messages = []
            for msg in messages:
                formatted_msg = {
                    'id': str(msg.get('id', '')),
                    'from': str(msg.get('from', 'Unknown')),
                    'subject': str(msg.get('subject', 'No Subject')),
                    'date': self._format_date(msg.get('date')),
                    'body': str(msg.get('body', '')),
                    'html': str(msg.get('html', '')) if msg.get('html') else None,
                    'attachments': msg.get('attachments', [])
                }
                formatted_messages.append(formatted_msg)

            return formatted_messages

        except Exception as e:
            self.logger.error(f"Error getting messages for {email}: {str(e)}")
            raise

    def _format_date(self, date_value) -> str:
        """Format date value to string, handling various formats"""
        if not date_value:
            return 'Unknown'
            
        try:
            if isinstance(date_value, (int, float)):
                # Unix timestamp
                return datetime.fromtimestamp(int(date_value)).strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(date_value, str):
                # Try parsing as timestamp first
                try:
                    return datetime.fromtimestamp(int(date_value)).strftime('%Y-%m-%d %H:%M:%S')
                except (ValueError, TypeError):
                    # Try parsing as ISO format
                    return datetime.fromisoformat(date_value).strftime('%Y-%m-%d %H:%M:%S')
            else:
                return str(date_value)
        except Exception as e:
            self.logger.warning(f"Could not parse date {date_value}: {str(e)}")
            return str(date_value)

    def get_message(self, email: str, message_id: str) -> Optional[Dict]:
        """Get a specific message"""
        try:
            provider = self.get_provider_for_email(email)
            if not provider:
                # Try getting provider by domain
                domain = email.split('@')[1]
                provider = self.get_provider_by_domain(domain)
        
            if provider:
                try:
                    message = provider.get_message(email, message_id)
                    if not message:
                        self.logger.info(f"Message {message_id} not found for {email}")
                        return None
                    
                    if not isinstance(message, dict):
                        self.logger.error(f"Invalid message format for {email}: not a dict")
                        return None
                
                    # Ensure all required fields exist
                    message['from'] = message.get('from', 'Unknown')
                    message['subject'] = message.get('subject', 'No Subject')
                    message['date'] = message.get('date', 'Unknown')
                    message['body'] = message.get('body', '')
                
                    return message
                
                except Exception as e:
                    self.logger.error(f"Error getting message from provider: {str(e)}")
                    return None
            return None
        except Exception as e:
            self.logger.error(f"Error getting message: {str(e)}")
            return None

    def generate_new_email(self) -> Optional[str]:
        """Generate a new temporary email address"""
        try:
            # Initialize providers if not already done
            if not self.providers:
                self.providers = {
                    'guerrilla': GuerrillaMailAPI(),
                    'tempmail': TempMailOrgAPI(),
                    'disposable': DisposableMailAPI(),
                    'yopmail': YopMailAPI()
                }
            
            # Try each provider in order until one succeeds
            for provider_name, provider in self.providers.items():
                try:
                    email = provider.generate_email()
                    if email:
                        # Add to managed emails
                        self.add_email(email, provider)
                        return email
                except Exception as e:
                    self.logger.warning(f"Failed to generate email with {provider_name}: {str(e)}")
                    continue
            
            raise EmailGenerationError("All providers failed to generate email")
            
        except Exception as e:
            self.logger.error(f"Error generating new email: {str(e)}")
            return None

    def get_available_domains(self) -> List[str]:
        """Get list of available domains"""
        domains = []
        for provider in [GuerrillaMailAPI(), MailTmAPI(), TempMailNinjaAPI(), DisposableMailAPI(), YopMailAPI(), TempMailOrgAPI()]:
            domains.extend(provider.get_available_domains())
        return domains

    def get_advanced_analytics(self) -> Dict:
        """Get advanced analytics about email usage"""
        analytics = {
            'total_emails': len(self.emails),
            'total_messages': sum(len(data.get('messages', [])) for data in self.emails.values()),
            'emails_by_domain': {},
            'messages_by_date': {},
            'avg_response_time': 0,
            'most_common_senders': {},
            'most_common_subjects': {},
        }
        
        total_response_time = 0
        response_count = 0
        
        for email_data in self.emails.values():
            # Domain analytics
            domain = email_data['email'].split('@')[1]
            analytics['emails_by_domain'][domain] = analytics['emails_by_domain'].get(domain, 0) + 1
            
            # Message analytics
            for msg in email_data.get('messages', []):
                # Date analytics
                date = msg['received_at'].split(' ')[0]
                analytics['messages_by_date'][date] = analytics['messages_by_date'].get(date, 0) + 1
                
                # Sender analytics
                sender = msg['from']
                analytics['most_common_senders'][sender] = analytics['most_common_senders'].get(sender, 0) + 1
                
                # Subject analytics
                subject = msg['subject']
                analytics['most_common_subjects'][subject] = analytics['most_common_subjects'].get(subject, 0) + 1
                
                # Response time analytics
                if msg.get('response_time'):
                    total_response_time += msg['response_time']
                    response_count += 1
        
        # Calculate average response time
        if response_count > 0:
            analytics['avg_response_time'] = total_response_time / response_count
            
        # Sort dictionaries by value
        analytics['most_common_senders'] = dict(sorted(
            analytics['most_common_senders'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5])
        
        analytics['most_common_subjects'] = dict(sorted(
            analytics['most_common_subjects'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5])
        
        return analytics

    def export_data(self, format='json') -> str:
        """Export email data to various formats"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'json':
            export_file = f'email_export_{timestamp}.json'
            with open(export_file, 'w') as f:
                json.dump(self.emails, f, indent=2)
            return export_file
            
        elif format == 'csv':
            export_file = f'email_export_{timestamp}.csv'
            
            # Flatten the data structure
            rows = []
            headers = ['email', 'created_at', 'message_id', 'subject', 'from', 'received_at', 'body']
            
            for email_data in self.emails.values():
                base_data = {
                    'email': email_data['email'],
                    'created_at': email_data['created_at']
                }
                
                if email_data.get('messages'):
                    for msg in email_data['messages']:
                        row = base_data.copy()
                        row.update({
                            'message_id': msg.get('id', ''),
                            'subject': msg.get('subject', ''),
                            'from': msg.get('from', ''),
                            'received_at': msg.get('received_at', ''),
                            'body': msg.get('body', '').replace('\n', ' ')
                        })
                        rows.append(row)
                else:
                    rows.append(base_data)
            
            # Write to CSV
            import csv
            with open(export_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)
            
            return export_file
        
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def generate_test_data(self, num_emails=5, messages_per_email=3):
        """Generate test data for development and testing"""
        fake = Faker()
        
        for _ in range(num_emails):
            email = self.generate_new_email()
            if not email:
                continue
                
            # Generate fake messages
            for _ in range(messages_per_email):
                message_data = {
                    'id': str(uuid.uuid4()),
                    'subject': fake.sentence(),
                    'from': fake.email(),
                    'body': fake.paragraph(),
                    'html': f"<p>{fake.paragraph()}</p>",
                    'received_at': fake.date_time_between(
                        start_date='-1d',
                        end_date='now'
                    ).strftime('%Y-%m-%d %H:%M:%S'),
                    'attachments': [],
                    'is_read': False,
                    'tags': [],
                    'notes': '',
                    'extracted_otp': None
                }
                
                # Randomly add OTP to some messages
                if random.random() < 0.3:  # 30% chance
                    otp = ''.join(random.choices(string.digits, k=6))
                    message_data['body'] += f"\nYour OTP is: {otp}"
                    message_data['extracted_otp'] = otp
                
                # Find the email data and append the message
                email_data = next((data for data in self.emails.values() if data['email'] == email), None)
                if email_data:
                    email_data['messages'].append(message_data)
        
        self.save_data()
        return True

    def forward_email(self, from_email: str, to_email: str, message_id: str) -> bool:
        """
        Forward an email from one address to another
        
        Args:
            from_email (str): Source email address
            to_email (str): Target email address (can be any valid email)
            message_id (str): ID of the message to forward
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.logger.info(f"Starting to forward message {message_id} from {from_email} to {to_email}")
            
            # Validate email addresses
            if not self.validate_email(from_email) or not self.validate_email(to_email):
                self.logger.error("Invalid email address format")
                return False
            
            # Get provider for source email
            provider = self.get_provider_for_email(from_email)
            if not provider:
                self.logger.error(f"No provider found for {from_email}")
                return False
                
            # Get message to forward
            self.logger.info(f"Fetching message {message_id} from {from_email}")
            message = provider.get_message(from_email, message_id)
            if not message:
                self.logger.error(f"Message {message_id} not found")
                return False
                
            self.logger.info(f"Message retrieved successfully: {message.get('subject', 'No Subject')}")
                
            # Check if target is a temporary email
            target_provider = self.get_provider_for_email(to_email)
            if target_provider:
                self.logger.info(f"Using provider {target_provider.__class__.__name__} for forwarding")
                # Use provider's forward_message for temp emails
                message['forwarded_from'] = from_email
                success = target_provider.forward_message(to_email, message)
                if success:
                    self.logger.info("Message forwarded successfully using provider")
                else:
                    self.logger.error("Provider failed to forward message")
                return success
            else:
                self.logger.info(f"Forwarding to external email {to_email} via SMTP")
                # For external emails, use SMTP
                success = self._forward_to_external_email(from_email, to_email, message)
                if success:
                    self.logger.info("Message forwarded successfully via SMTP")
                else:
                    self.logger.error("SMTP forwarding failed")
                return success
                
        except Exception as e:
            self.logger.error(f"Error forwarding email: {str(e)}", exc_info=True)
            return False
            
    def _forward_to_external_email(self, from_email: str, to_email: str, message: Dict) -> bool:
        """Forward a message to an external email address using SMTP"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            from email.mime.application import MIMEApplication
            from email.utils import formatdate
            from bs4 import BeautifulSoup
            import base64
            
            # Get SMTP settings
            smtp_host = os.getenv('SMTP_HOST')
            smtp_port = int(os.getenv('SMTP_PORT', '587'))
            smtp_user = os.getenv('SMTP_USER')
            smtp_pass = os.getenv('SMTP_PASS')
            
            if not all([smtp_host, smtp_port, smtp_user, smtp_pass]):
                self.logger.error("Missing SMTP settings")
                return False
            
            # Create message
            msg = MIMEMultipart('mixed')
            msg['Subject'] = f"Fwd: {message.get('subject', 'No Subject')}"
            msg['From'] = smtp_user
            msg['To'] = to_email
            msg['Date'] = formatdate(localtime=True)
            
            # Create the body as a multipart alternative (plain text + HTML)
            body = MIMEMultipart('alternative')
            
            # Create forwarding header
            forward_header = (
                "\n\n---------- Forwarded message ----------\n"
                f"From: {message.get('from', 'Unknown')}\n"
                f"Date: {self._format_date(message.get('date'))}\n"
                f"Subject: {message.get('subject', 'No Subject')}\n"
                f"To: {to_email}\n\n"
            )
            
            # Get message content
            content = message.get('body', '')
            html_content = message.get('html')
            
            # If content contains HTML but html_content is None, use content as HTML
            if html_content is None and ('<html' in content.lower() or '<body' in content.lower() or '<div' in content.lower()):
                html_content = content
                # Create plain text from HTML
                soup = BeautifulSoup(html_content, 'html.parser')
                content = soup.get_text()
            
            # Create plain text part
            text_part = forward_header + content
            body.attach(MIMEText(text_part, 'plain', 'utf-8'))
            
            # Create HTML part
            if html_content:
                html_header = forward_header.replace('\n', '<br>')
                full_html = f"""
                <html>
                    <head>
                        <style>
                            .forwarded-message {{
                                margin: 20px 0;
                                padding: 10px;
                                border-left: 2px solid #ccc;
                                color: #666;
                            }}
                            .message-content {{
                                margin-top: 20px;
                            }}
                        </style>
                    </head>
                    <body>
                        <div class="forwarded-message">
                            {html_header}
                        </div>
                        <div class="message-content">
                            {html_content}
                        </div>
                    </body>
                </html>
                """
            else:
                # Convert plain text to simple HTML
                escaped_content = content.replace('\n', '<br>')
                full_html = f"""
                <html>
                    <head>
                        <style>
                            .forwarded-message {{
                                margin: 20px 0;
                                padding: 10px;
                                border-left: 2px solid #ccc;
                                color: #666;
                            }}
                            .message-content {{
                                margin-top: 20px;
                                white-space: pre-wrap;
                            }}
                        </style>
                    </head>
                    <body>
                        <div class="forwarded-message">
                            {forward_header.replace('\n', '<br>')}
                        </div>
                        <div class="message-content">
                            {escaped_content}
                        </div>
                    </body>
                </html>
                """
            
            body.attach(MIMEText(full_html, 'html', 'utf-8'))
            msg.attach(body)
            
            # Add attachments if any
            attachments = message.get('attachments', [])
            if attachments:
                for attachment in attachments:
                    try:
                        filename = attachment.get('filename', 'attachment.dat')
                        content = attachment.get('content')
                        
                        if content:
                            # Handle base64 encoded content if needed
                            if isinstance(content, str):
                                try:
                                    content = base64.b64decode(content)
                                except:
                                    content = content.encode()
                            
                            # Create attachment part
                            att = MIMEApplication(content)
                            att.add_header(
                                'Content-Disposition',
                                'attachment',
                                filename=filename
                            )
                            msg.attach(att)
                            self.logger.info(f"Added attachment: {filename}")
                    except Exception as e:
                        self.logger.error(f"Failed to add attachment {filename}: {str(e)}")
            
            # Connect and send
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error forwarding email: {str(e)}")
            return False

    def validate_email(self, email: str) -> bool:
        """
        Validate email format and check if it's from a supported provider
        
        Args:
            email (str): Email address to validate
        
        Returns:
            bool: True if email is valid, False otherwise
        """
        try:
            # Basic format validation using a more comprehensive regex
            email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_regex, email):
                self.logger.error(f"Invalid email format: {email}")
                return False
                
            # Extract domain
            domain = email.split('@')[-1].lower()
            
            # For source emails, check if domain is supported by any provider
            supported_domains = set()
            for provider in [GuerrillaMailAPI(), DisposableMailAPI(), 
                            YopMailAPI(), TempMailOrgAPI()]:
                supported_domains.update(provider.get_available_domains())
            
            # If email is in our providers, it must be from a supported domain
            if any(email.lower().endswith('@' + d) for d in supported_domains):
                return True
                
            # For external email addresses (like Gmail), just validate the format
            return True
            
        except Exception as e:
            self.logger.error(f"Email validation error: {str(e)}")
            return False

    def get_provider_by_domain(self, email: str) -> Optional[TempMailAPI]:
        """
        Get the email provider instance for a given email address based on its domain
        
        Args:
            email (str): Email address
            
        Returns:
            Optional[TempMailAPI]: Provider instance if found, None otherwise
        """
        try:
            domain = email.split('@')[-1].lower()
            
            # Check each provider's supported domains
            for provider in [GuerrillaMailAPI(), DisposableMailAPI(), 
                           YopMailAPI(), TempMailOrgAPI()]:
                if domain in provider.get_available_domains():
                    return provider
                    
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting provider for email: {str(e)}")
            return None

def main():
    """Example usage of TempMailManager"""
    mail_manager = TempMailManager()
    
    try:
        # Generate new email
        print("\nGenerating new temporary email...")
        new_email = mail_manager.generate_new_email()
        
        if new_email:
            print(f"\nGenerated email: {new_email}")
            
            # Start monitoring the email
            print("\nStarting email monitoring...")
            mail_manager.start_monitoring(new_email)
            
            # Wait for and check messages
            print("\nWaiting for messages (press Ctrl+C to stop)...")
            try:
                while True:
                    messages = mail_manager.check_messages(new_email)
                    if messages:
                        print("\nMessages received:")
                        for msg in messages:
                            print(f"\nFrom: {msg['from']}")
                            print(f"Subject: {msg['subject']}")
                            print(f"Time: {msg['received_at']}")
                            if msg['extracted_otp']:
                                print(f"OTP found: {msg['extracted_otp']}")
                            print("-" * 50)
                    time.sleep(5)  # Check every 5 seconds
                    
            except KeyboardInterrupt:
                print("\nStopping email monitoring...")
                mail_manager.stop_monitoring(new_email)
                
    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        print("\nCleaning up...")
        time.sleep(6)  # Wait for 6 seconds
        os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen

if __name__ == "__main__":
    main()

class YopMailAPI(TempMailAPI):
    """API client for YopMail"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.base_url = "https://yopmail.com"
        
    def get_available_domains(self) -> List[str]:
        return ["yopmail.com", "yopmail.net", "cool.fr.nf", "jetable.fr.nf", 
                "nospam.ze.tc", "nomail.xl.cx", "mega.zik.dj", "speed.1s.fr"]
                
    def generate_email(self) -> str:
        try:
            username = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
            domain = random.choice(self.get_available_domains())
            return f"{username}@{domain}"
        except Exception as e:
            raise ValueError(f"Failed to generate YopMail address: {str(e)}")

    def get_messages(self, email: str) -> List[Dict]:
        try:
            login = email.split('@')[0]
            response = self.session.get(f"{self.base_url}/inbox/{login}")
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            messages = []
            for msg in soup.select('.m'):
                msg_id = msg.get('id')
                if msg_id:
                    messages.append({
                        'id': msg_id,
                        'subject': msg.select_one('.lms').text if msg.select_one('.lms') else 'No Subject',
                        'from': msg.select_one('.lmf').text if msg.select_one('.lmf') else 'Unknown',
                        'date': msg.select_one('.lmd').text if msg.select_one('.lmd') else 'Unknown'
                    })
            return messages
        except Exception as e:
            logging.error(f"Error getting messages: {str(e)}")
            return []

    def get_message(self, email: str, message_id: str) -> Optional[Dict]:
        try:
            login = email.split('@')[0]
            response = self.session.get(f"{self.base_url}/inbox/{login}/{message_id}")
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            body = soup.select_one('#mail')
            return {
                'id': message_id,
                'subject': soup.select_one('.hm').text if soup.select_one('.hm') else 'No Subject',
                'from': soup.select_one('.lmf').text if soup.select_one('.lmf') else 'Unknown',
                'date': soup.select_one('.lmd').text if soup.select_one('.lmd') else 'Unknown',
                'body': body.get_text() if body else ''
            }
        except Exception as e:
            logging.error(f"Error getting message: {str(e)}")
            return None

    def get_provider_name(self) -> str:
        return "YopMail"

class TempMailOrgAPI(TempMailAPI):
    """API client for Temp-Mail.org"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.base_url = "https://web2.temp-mail.org"
        
    def get_available_domains(self) -> List[str]:
        try:
            response = self.session.get(f"{self.base_url}/api/v3/domains")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Error getting domains: {str(e)}")
            return ["temp-mail.org", "temp-mail.com"]

    def generate_email(self) -> str:
        try:
            username = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
            domains = self.get_available_domains()
            domain = random.choice(domains)
            return f"{username}@{domain}"
        except Exception as e:
            raise ValueError(f"Failed to generate Temp-Mail.org address: {str(e)}")

    def get_messages(self, email: str) -> List[Dict]:
        try:
            response = self.session.get(f"{self.base_url}/api/v3/email/{email}/messages")
            response.raise_for_status()
            return [{
                'id': str(msg['id']),
                'subject': msg.get('subject', 'No Subject'),
                'from': msg.get('from', 'Unknown'),
                'date': msg.get('created_at', 'Unknown')
            } for msg in response.json()]
        except Exception as e:
            logging.error(f"Error getting messages: {str(e)}")
            return []

    def get_message(self, email: str, message_id: str) -> Optional[Dict]:
        try:
            response = self.session.get(f"{self.base_url}/api/v3/email/{email}/messages/{message_id}")
            response.raise_for_status()
            msg = response.json()
            return {
                'id': str(msg['id']),
                'subject': msg.get('subject', 'No Subject'),
                'from': msg.get('from', 'Unknown'),
                'date': msg.get('created_at', 'Unknown'),
                'body': msg.get('body', '')
            }
        except Exception as e:
            logging.error(f"Error getting message: {str(e)}")
            return None

    def get_provider_name(self) -> str:
        return "Temp-Mail.org"

class DisposableMailAPI(TempMailAPI):
    """API client for 1secmail"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.base_url = "https://www.1secmail.net"
        
    def get_available_domains(self) -> List[str]:
        try:
            response = self.session.get(f"{self.base_url}/api/v1/?action=getDomainList")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Error getting domains: {str(e)}")
            return ["1secmail.com", "1secmail.org", "1secmail.net"]

    def generate_email(self) -> str:
        try:
            username = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
            domain = random.choice(self.get_available_domains())
            return f"{username}@{domain}"
        except Exception as e:
            raise ValueError(f"Failed to generate 1secmail address: {str(e)}")

    def get_messages(self, email: str) -> List[Dict]:
        try:
            login, domain = email.split('@')
            response = self.session.get(
                f"{self.base_url}/api/v1/",
                params={
                    "action": "getMessages",
                    "login": login,
                    "domain": domain
                }
            )
            response.raise_for_status()
            return [{
                'id': str(msg['id']),
                'subject': msg.get('subject', 'No Subject'),
                'from': msg.get('from', 'Unknown'),
                'date': msg.get('date', 'Unknown')
            } for msg in response.json()]
        except Exception as e:
            logging.error(f"Error getting messages: {str(e)}")
            return []

    def get_message(self, email: str, message_id: str) -> Optional[Dict]:
        try:
            login, domain = email.split('@')
            response = self.session.get(
                f"{self.base_url}/api/v1/",
                params={
                    "action": "readMessage",
                    "login": login,
                    "domain": domain,
                    "id": message_id
                }
            )
            response.raise_for_status()
            msg = response.json()
            return {
                'id': str(msg['id']),
                'subject': msg.get('subject', 'No Subject'),
                'from': msg.get('from', 'Unknown'),
                'date': msg.get('date', 'Unknown'),
                'body': msg.get('body', '')
            }
        except Exception as e:
            logging.error(f"Error getting message: {str(e)}")
            return None

    def get_provider_name(self) -> str:
        return "1secmail"

class GuerrillaMailAPI(TempMailAPI):
    """Guerrilla Mail provider"""
    def __init__(self):
        """Initialize provider"""
        self.base_url = 'https://api.guerrillamail.com/ajax.php'
        self.session_id = None
        self.domains = [
            "guerrillamail.com",
            "guerrillamail.net",
            "guerrillamail.org",
            "grr.la",
            "sharklasers.com"
        ]
        self.session = retry_with_backoff()
        self._init_session()

    def generate_random_username(self) -> str:
        """Generate a random username with consistent pattern"""
        username = ''.join(random.choices(string.ascii_lowercase, k=4))
        username += ''.join(random.choices(string.digits, k=4))
        username += ''.join(random.choices(string.ascii_lowercase, k=4))
        return username

    def _init_session(self):
        """Initialize Guerrilla Mail session"""
        try:
            response = self.session.get(
                f"{self.base_url}",
                params={
                    "f": "get_email_address",
                    "ip": "127.0.0.1",
                    "agent": "Mozilla_foo_bar"
                }
            )
            response.raise_for_status()
            data = response.json()
            self.session_id = data.get('sid_token')
            logging.info(f"Initialized Guerrilla Mail session: {self.session_id}")
        except Exception as e:
            logging.error(f"Error initializing session: {str(e)}")
            raise APIError("Failed to initialize Guerrilla Mail session")

    def _check_session(self):
        """Check and reinitialize session if needed"""
        if not self.session_id:
            self._init_session()

    def _set_email_address(self, email_user: str, domain: str):
        """Set the current email address"""
        try:
            response = self.session.get(
                f"{self.base_url}",
                params={
                    "f": "set_email_user",
                    "email_user": email_user,
                    "domain": domain,
                    "sid_token": self.session_id
                }
            )
            response.raise_for_status()
            data = response.json()
            logging.info(f"Set email address: {email_user}@{domain}")
            return True
        except Exception as e:
            logging.error(f"Error setting email address: {str(e)}")
            return False

    def _forget_me(self):
        """Reset session"""
        try:
            response = self.session.get(
                f"{self.base_url}",
                params={
                    "f": "forget_me",
                    "sid_token": self.session_id
                }
            )
            response.raise_for_status()
            self.session_id = None
            self._init_session()
        except Exception as e:
            logging.error(f"Error resetting session: {str(e)}")

    def generate_email(self) -> str:
        """Generate a new email address"""
        try:
            self._check_session()
            
            # Generate random username with consistent pattern
            username = self.generate_random_username()
            
            # Use most reliable domain
            domain = "sharklasers.com"
            
            # Set email address
            if not self._set_email_address(username, domain):
                raise EmailGenerationError("Failed to set email address")
            
            # Verify session is active
            response = self.session.get(
                f"{self.base_url}",
                params={
                    'f': 'get_email_address',
                    'sid_token': self.session_id
                },
                timeout=10
            )
            response.raise_for_status()
            
            email = f"{username}@{domain}"
            return email
            
        except Exception as e:
            logging.error(f"Error generating email: {str(e)}")
            self._forget_me()  # Reset session on error
            raise EmailGenerationError("Failed to generate Guerrilla Mail address")

    def get_messages(self, email: str) -> List[Dict]:
        """Get messages for an email address"""
        try:
            self._check_session()
            
            email_user = email.split('@')[0]
            domain = email.split('@')[1]
            
            # Set email address
            if not self._set_email_address(email_user, domain):
                return []
            
            # Get messages with retries
            for attempt in range(3):
                try:
                    # Get message list
                    response = self.session.get(
                        f"{self.base_url}",
                        params={
                            "f": "get_email_list",
                            "offset": 0,
                            "sid_token": self.session_id
                        }
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    # Extract messages
                    messages = []
                    for msg in data.get('list', []):
                        messages.append({
                            'id': str(msg.get('mail_id')),
                            'from': msg.get('mail_from', 'Unknown'),
                            'subject': msg.get('mail_subject', 'No Subject'),
                            'date': msg.get('mail_timestamp', 'Unknown'),
                            'body': msg.get('mail_excerpt', '')
                        })
                    
                    return messages
                    
                except Exception as e:
                    logging.error(f"Error on attempt {attempt + 1}: {str(e)}")
                    if attempt == 2:  # Last attempt
                        self._forget_me()  # Reset session
                    else:
                        sleep(2)  # Wait before retry
            
            return []
            
        except Exception as e:
            logging.error(f"Error getting messages: {str(e)}")
            return []

    def get_message(self, email: str, message_id: str) -> Optional[Dict]:
        """Get a specific message"""
        try:
            self._check_session()
            
            email_user = email.split('@')[0]
            domain = email.split('@')[1]
            
            # Set email address
            if not self._set_email_address(email_user, domain):
                return None
            
            # Get message
            response = self.session.get(
                f"{self.base_url}",
                params={
                    "f": "fetch_email",
                    "email_id": message_id,
                    "sid_token": self.session_id
                }
            )
            response.raise_for_status()
            msg = response.json()
            
            # Extract message content
            return {
                'id': str(msg.get('mail_id')),
                'from': msg.get('mail_from', 'Unknown'),
                'subject': msg.get('mail_subject', 'No Subject'),
                'date': msg.get('mail_timestamp', 'Unknown'),
                'body': msg.get('mail_body', ''),
                'html': msg.get('mail_html', None)
            }
            
        except Exception as e:
            logging.error(f"Error getting message: {str(e)}")
            return None

    def get_provider_name(self) -> str:
        """Get provider name"""
        return "GuerrillaMailAPI"

    def get_available_domains(self) -> List[str]:
        """Get list of available domains"""
        return self.domains

    def send_message(self, message_data: Dict) -> bool:
        """Send a message using Guerrilla Mail API"""
        try:
            self._check_session()
            
            # Set email address for sending
            email_user = message_data['to'].split('@')[0]
            domain = message_data['to'].split('@')[1]
            if not self._set_email_address(email_user, domain):
                return False
            
            # Format message data
            data = {
                'to': message_data['to'],
                'subject': message_data['subject'],
                'from': message_data.get('from', ''),
                'message': message_data.get('html', '') or message_data.get('body', ''),
                'sid_token': self.session_id,
                'email': message_data.get('from', ''),  # Required by API
                'in_reply_to': ''  # Not a reply
            }
            
            # Send message
            response = self.session.post(
                f"{self.base_url}",
                params={'f': 'send_email'},
                data=data,
                timeout=30
            )
            response.raise_for_status()
            
            # Check response
            result = response.json()
            if result.get('status') == 'success' or 'mail_id' in result:
                # Wait briefly for message to be delivered
                sleep(1)
                
                # Verify message appears in inbox
                messages = self.get_messages(message_data['to'])
                for msg in messages:
                    if msg.get('subject') == message_data['subject']:
                        return True
                        
                return True  # Return true even if verification fails
            else:
                logging.error(f"Failed to send message: {result.get('message', 'Unknown error')}")
                return False
                
        except Exception as e:
            logging.error(f"Error sending message: {str(e)}")
            return False

    def forward_message(self, to_email: str, message_data: Dict) -> bool:
        """Forward a message using Guerrilla Mail API"""
        try:
            self._check_session()
            
            # Get the full message content if needed
            if 'html' not in message_data or not message_data.get('body'):
                message_id = message_data.get('id')
                if message_id:
                    full_msg = self.get_message(to_email, message_id)
                    if full_msg:
                        message_data.update(full_msg)
            
            # Format message for forwarding
            forward_data = {
                'to': to_email,
                'subject': f"Fwd: {message_data.get('subject', 'No Subject')}",
                'from': message_data.get('forwarded_from', to_email),  # Use destination email as sender
                'body': message_data.get('body', ''),
                'html': message_data.get('html', None)
            }
            
            # Create forwarding header
            forward_header = (
                "\n\n---------- Forwarded message ----------\n"
                f"From: {message_data.get('from', 'Unknown')}\n"
                f"Date: {message_data.get('date', 'Unknown')}\n"
                f"Subject: {message_data.get('subject', 'No Subject')}\n"
                f"To: {to_email}\n\n"
            )
            
            # Add headers to content
            if forward_data.get('html'):
                html_header = forward_header.replace('\n', '<br>')
                forward_data['html'] = f"<div style='white-space: pre-wrap;'>{html_header}</div>{forward_data['html']}"
            else:
                forward_data['body'] = forward_header + forward_data['body']
            
            # Send the forwarded message
            success = self.send_message(forward_data)
            if success:
                # Wait briefly for the message to appear
                sleep(2)
                return True
            return False
                
        except Exception as e:
            logging.error(f"Error forwarding message: {str(e)}")
            return False

__all__ = [
    'TempMailAPI',
    'TempMailManager',
    'GuerrillaMailAPI',
    'DisposableMailAPI',
    'YopMailAPI',
    'TempMailOrgAPI',
    'EmailMessage',
    'EmailForwarder',
    'RateLimiter',
    'EmailTemplate',
    'ProviderConfig',
    'TenMinuteMailConfig',
    'DisposableMailConfig',
    'MohmalConfig',
    'EmailProviderError',
    'EmailGenerationError',
    'MessageFetchError',
    'AuthenticationError',
    'APIError'
]
