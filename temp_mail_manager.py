import os
import re
import uuid
import json
import queue
import string
import random
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from threading import Lock
from time import time, sleep
from hashlib import md5
from base64 import b64decode
from mimetypes import guess_type
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from aiosmtplib import SMTP
from faker import Faker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger('temp_mail_manager')

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
    """Rate limiter using token bucket algorithm"""
    def __init__(self, tokens_per_second: float, burst_size: int):
        self.tokens_per_second = tokens_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time()
        self.lock = Lock()

    def acquire(self, tokens: int = 1) -> bool:
        with self.lock:
            now = time()
            time_passed = now - self.last_update
            self.tokens = min(
                self.burst_size,
                self.tokens + time_passed * self.tokens_per_second
            )
            self.last_update = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

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

class RateLimiter:
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.lock = Lock()

    def can_proceed(self) -> bool:
        current_time = time()
        with self.lock:
            # Remove old requests
            self.requests = [req_time for req_time in self.requests 
                           if current_time - req_time < self.time_window]
            
            if len(self.requests) < self.max_requests:
                self.requests.append(current_time)
                return True
            return False

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
        self.emails = {}
        self.data_file = os.path.expanduser('~/.tempmail/emails.json')
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
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
                logger.info(f"Added email {email} with provider {provider_class}")
                self.save_data()
                return True
            return False
        except Exception as e:
            logger.error(f"Error adding email: {str(e)}")
            return False

    def get_provider_for_email(self, email: str) -> Optional[TempMailAPI]:
        """Get provider instance for an email"""
        try:
            if email in self.emails:
                provider_class = self.emails[email].get('provider_class')
                logger.info(f"Getting provider {provider_class} for email {email}")
                
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
            logger.error(f"Error getting provider for email: {str(e)}")
            return None

    def get_active_emails(self) -> List[Dict]:
        """Get list of active emails"""
        try:
            return [
                {
                    'email': email,
                    'provider': data['provider_class'],
                    'created_at': data['created_at']
                }
                for email, data in self.emails.items()
            ]
        except Exception as e:
            logger.error(f"Error getting active emails: {str(e)}")
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
            logger.error(f"Error deleting email: {str(e)}")
            return False

    def load_data(self):
        """Load saved email data"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    self.emails = json.load(f)
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            self.emails = {}

    def save_data(self):
        """Save email data"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.emails, f, indent=4)
            logger.info("Email data saved successfully")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")

    def get_messages(self, email: str) -> List[Dict]:
        """Get messages for an email"""
        try:
            provider = self.get_provider_for_email(email)
            if not provider:
                # Try getting provider by domain
                domain = email.split('@')[1]
                provider = self.get_provider_by_domain(domain)
                if provider:
                    # Update stored provider
                    self.emails[email] = {
                        'provider_class': provider.__class__.__name__,
                        'created_at': self.emails.get(email, {}).get('created_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    }
                    self.save_data()
        
            if provider:
                try:
                    messages = provider.get_messages(email)
                    if not messages:
                        logger.info(f"No messages found for {email}")
                        return []
                    
                    if not isinstance(messages, list):
                        logger.error(f"Invalid message format for {email}: not a list")
                        return []
                
                    # Process and validate each message
                    valid_messages = []
                    for msg in messages:
                        if not isinstance(msg, dict):
                            continue
                        
                        msg_id = msg.get('id')
                        if not msg_id:
                            continue
                        
                        # Ensure all required fields exist
                        msg['from'] = msg.get('from', 'Unknown')
                        msg['subject'] = msg.get('subject', 'No Subject')
                        msg['date'] = msg.get('date', 'Unknown')
                        msg['body'] = msg.get('body', '')
                    
                        valid_messages.append(msg)
                
                    return valid_messages
                
                except Exception as e:
                    logger.error(f"Error getting messages from provider: {str(e)}")
                    return []
            return []
        except Exception as e:
            logger.error(f"Error getting messages: {str(e)}")
            return []

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
                        logger.info(f"Message {message_id} not found for {email}")
                        return None
                    
                    if not isinstance(message, dict):
                        logger.error(f"Invalid message format for {email}: not a dict")
                        return None
                
                    # Ensure all required fields exist
                    message['from'] = message.get('from', 'Unknown')
                    message['subject'] = message.get('subject', 'No Subject')
                    message['date'] = message.get('date', 'Unknown')
                    message['body'] = message.get('body', '')
                
                    return message
                
                except Exception as e:
                    logger.error(f"Error getting message from provider: {str(e)}")
                    return None
            return None
        except Exception as e:
            logger.error(f"Error getting message: {str(e)}")
            return None

    def generate_new_email(self) -> Optional[str]:
        """Generate a new temporary email address"""
        try:
            # Generate random username
            username = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
            
            # Get available domains
            domains = self.get_available_domains()
            if not domains:
                logger.error("No domains available")
                return None
                
            # Select random domain
            domain = random.choice(domains)
            
            # Create email
            email = f"{username}@{domain}"
            
            # Validate email
            if not self.validate_email(email):
                logger.error(f"Generated invalid email: {email}")
                return None
                
            return email
            
        except Exception as e:
            logger.error(f"Error generating email: {str(e)}")
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
        """Forward an email from one temp mail to another"""
        try:
            # Get source provider
            from_provider = self.get_provider(from_email)
            if not from_provider:
                raise ValueError(f"No provider found for source email {from_email}")

            # Get target provider
            to_provider = self.get_provider(to_email)
            if not to_provider:
                raise ValueError(f"No provider found for target email {to_email}")

            # Get the message content
            message = None
            for email_data in self.emails.values():
                if email_data['email'] == from_email:
                    for msg in email_data.get('messages', []):
                        if msg.get('id') == message_id:
                            message = msg
                            break
                    break

            if not message:
                raise ValueError(f"Message {message_id} not found")

            # Create forwarded message body
            forward_body = self._create_forward_body(message, message)

            # Forward the message using the target provider
            success = to_provider.send_test_email(
                email=to_email,
                subject=f"Fwd: {message.get('subject', 'No Subject')}",
                body=forward_body
            )

            if success:
                logger.info(f"Successfully forwarded message {message_id} from {from_email} to {to_email}")
                return True
            else:
                logger.error(f"Failed to forward message {message_id}")
                return False

        except Exception as e:
            logger.error(f"Error forwarding email: {str(e)}")
            return False

    def _create_forward_body(self, original_message: Dict, message_content: Dict) -> str:
        """Create the forwarded message body with proper formatting"""
        # Get message details
        from_addr = original_message.get('from', 'Unknown')
        date = datetime.fromtimestamp(int(original_message.get('date', 0))).strftime('%Y-%m-%d %H:%M:%S')
        subject = original_message.get('subject', 'No Subject')
        body = message_content.get('body', '')

        # Create forward header
        forward_header = f"""
---------- Forwarded message ----------
From: {from_addr}
Date: {date}
Subject: {subject}

"""
        return forward_header + body

    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        try:
            # Basic email format validation
            if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                return False
                
            # Check domain
            domain = email.split('@')[1]
            provider = self.get_provider(email)
            if not provider:
                return False
                
            # Check if domain is supported by provider
            domains = provider.get_available_domains()
            if domain not in domains:
                return False
                
            return True
        except Exception:
            return False

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
    """YopMail provider - Free and no auth required"""
    def __init__(self):
        """Initialize provider"""
        self.base_url = "https://yopmail.com/en"
        self.api_url = "https://yopmail.com/inbox"
        self.domains = [
            "yopmail.com",
            "yopmail.net",
            "yopmail.org",
            "cool.fr.nf",
            "jetable.fr.nf",
            "nospam.ze.tc",
            "nomail.xl.cx"
        ]
        self.session = retry_with_backoff()

    def generate_email(self) -> str:
        """Generate a new email address"""
        try:
            # Generate random username (10-12 chars)
            email_user = ''.join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(10, 12)))
            domain = random.choice(self.domains)
            
            # Verify email availability
            response = self.session.get(
                f"{self.base_url}/check.php",
                params={"login": email_user},
                headers={"User-Agent": "Mozilla/5.0"}
            )
            response.raise_for_status()
            
            return f"{email_user}@{domain}"
            
        except Exception as e:
            logger.error(f"Error generating YopMail address: {str(e)}")
            raise APIError("Failed to generate YopMail address")

    def get_messages(self, email: str) -> List[Dict]:
        """Get messages for an email address"""
        try:
            login = email.split('@')[0]
            
            # Get inbox
            response = self.session.get(
                f"{self.api_url}/{login}",
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "application/json"
                }
            )
            response.raise_for_status()
            
            # Parse HTML response
            soup = BeautifulSoup(response.text, 'html.parser')
            messages = []
            
            # Find message elements
            message_elements = soup.select('.m')
            for element in message_elements:
                try:
                    msg_id = element.get('id')
                    if msg_id:
                        msg = self.get_message(email, msg_id)
                        if msg:
                            messages.append(msg)
                except Exception as e:
                    logger.error(f"Error processing message {msg_id}: {str(e)}")
                    continue
            
            return messages
            
        except Exception as e:
            logger.error(f"Error getting messages from YopMail: {str(e)}")
            return []

    def get_message(self, email: str, message_id: str) -> Optional[Dict]:
        """Get a specific message"""
        try:
            login = email.split('@')[0]
            
            # Get message content
            response = self.session.get(
                f"{self.api_url}/{login}/{message_id}",
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "application/json"
                }
            )
            response.raise_for_status()
            
            # Parse HTML response
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract message details
            subject = soup.select_one('.hm').text.strip() if soup.select_one('.hm') else 'No Subject'
            sender = soup.select_one('.ellipsis').text.strip() if soup.select_one('.ellipsis') else 'Unknown'
            date = soup.select_one('.hm_date').text.strip() if soup.select_one('.hm_date') else 'Unknown'
            
            # Get message body
            body_element = soup.select_one('.mail')
            body = body_element.get_text(strip=True) if body_element else ''
            
            return {
                'id': message_id,
                'from': sender,
                'subject': subject,
                'date': date,
                'body': body
            }
            
        except Exception as e:
            logger.error(f"Error getting message from YopMail: {str(e)}")
            return None

    def get_provider_name(self) -> str:
        """Get provider name"""
        return "YopMail"

    def get_available_domains(self) -> List[str]:
        """Get list of available domains"""
        return self.domains


class TempMailOrgAPI(TempMailAPI):
    """Temp-Mail.org provider - Free and no auth required"""
    def __init__(self):
        """Initialize provider"""
        self.base_url = "https://temp-mail.org/api/v3"
        self.domains = [
            "temp-mail.org",
            "temp-mail.com",
            "tmpmail.org",
            "tmpmail.net",
            "tmails.net"
        ]
        self.session = retry_with_backoff()
        self._token = None

    def _get_token(self) -> Optional[str]:
        """Get authentication token"""
        try:
            if self._token:
                return self._token
                
            response = self.session.post(
                f"{self.base_url}/token",
                headers={"User-Agent": "Mozilla/5.0"}
            )
            response.raise_for_status()
            self._token = response.json().get('token')
            return self._token
            
        except Exception as e:
            logger.error(f"Error getting token: {str(e)}")
            return None

    def generate_email(self) -> str:
        """Generate a new email address"""
        try:
            # Generate random username
            email_user = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
            domain = random.choice(self.domains)
            
            # Get token
            token = self._get_token()
            if not token:
                raise APIError("Failed to get authentication token")
            
            # Register email
            response = self.session.post(
                f"{self.base_url}/email/new",
                headers={
                    "Authorization": f"Bearer {token}",
                    "User-Agent": "Mozilla/5.0"
                },
                json={"email": f"{email_user}@{domain}"}
            )
            response.raise_for_status()
            
            return f"{email_user}@{domain}"
            
        except Exception as e:
            logger.error(f"Error generating Temp-Mail.org address: {str(e)}")
            raise APIError("Failed to generate Temp-Mail.org address")

    def get_messages(self, email: str) -> List[Dict]:
        """Get messages for an email address"""
        try:
            # Get token
            token = self._get_token()
            if not token:
                return []
            
            # Get messages
            response = self.session.get(
                f"{self.base_url}/email/{email}/messages",
                headers={
                    "Authorization": f"Bearer {token}",
                    "User-Agent": "Mozilla/5.0"
                }
            )
            response.raise_for_status()
            messages_data = response.json()
            
            messages = []
            for msg in messages_data:
                try:
                    msg_id = msg.get('id')
                    if msg_id:
                        full_msg = self.get_message(email, str(msg_id))
                        if full_msg:
                            messages.append(full_msg)
                except Exception as e:
                    logger.error(f"Error processing message {msg.get('id')}: {str(e)}")
                    continue
            
            return messages
            
        except Exception as e:
            logger.error(f"Error getting messages from Temp-Mail.org: {str(e)}")
            return []

    def get_message(self, email: str, message_id: str) -> Optional[Dict]:
        """Get a specific message"""
        try:
            # Get token
            token = self._get_token()
            if not token:
                return None
            
            # Get message
            response = self.session.get(
                f"{self.base_url}/email/{email}/messages/{message_id}",
                headers={
                    "Authorization": f"Bearer {token}",
                    "User-Agent": "Mozilla/5.0"
                }
            )
            response.raise_for_status()
            msg = response.json()
            
            return {
                'id': str(msg.get('id')),
                'from': msg.get('from'),
                'subject': msg.get('subject'),
                'date': msg.get('created_at'),
                'body': msg.get('text_body', msg.get('html_body', ''))
            }
            
        except Exception as e:
            logger.error(f"Error getting message from Temp-Mail.org: {str(e)}")
            return None

    def get_provider_name(self) -> str:
        """Get provider name"""
        return "Temp-Mail.org"

    def get_available_domains(self) -> List[str]:
        """Get list of available domains"""
        return self.domains

class DisposableMailAPI(TempMailAPI):
    """1secmail.com provider - Free and no auth required"""
    def __init__(self):
        """Initialize provider"""
        self.base_url = "https://www.1secmail.com/api/v1/"
        self.domains = [
            "1secmail.com",  # Most reliable domain
            "1secmail.org",
            "1secmail.net"
        ]
        self.session = retry_with_backoff()

    def _validate_email(self, email: str) -> bool:
        """Validate email exists and is ready for use"""
        try:
            login, domain = email.split('@')
            response = self.session.get(
                f"{self.base_url}",
                params={
                    "action": "getMessages",
                    "login": login,
                    "domain": domain
                }
            )
            return response.status_code == 200
        except Exception:
            return False

    def generate_email(self) -> str:
        """Generate a new email address"""
        try:
            # Try to get email from API first
            response = self.session.get(
                f"{self.base_url}",
                params={
                    "action": "genRandomMailbox",
                    "count": 1
                }
            )
            response.raise_for_status()
            
            if response.json():
                email = response.json()[0]
                if self._validate_email(email):
                    return email

            # Fallback to manual generation if API fails
            for _ in range(3):  # Try up to 3 times
                # Use more reliable pattern for username
                email_user = ''.join(random.choices(string.ascii_lowercase, k=4))  # Start with letters
                email_user += ''.join(random.choices(string.digits, k=4))  # Add some numbers
                email_user += ''.join(random.choices(string.ascii_lowercase, k=4))  # End with letters
                
                domain = "1secmail.com"  # Use most reliable domain
                email = f"{email_user}@{domain}"
                
                if self._validate_email(email):
                    return email
                
                sleep(1)  # Wait before retry
            
            raise APIError("Failed to generate valid email address")
            
        except Exception as e:
            logger.error(f"Error generating 1secmail address: {str(e)}")
            raise APIError("Failed to generate 1secmail address")

    def get_messages(self, email: str) -> List[Dict]:
        """Get messages for an email address"""
        try:
            login, domain = email.split('@')
            
            # Verify email is valid
            if not self._validate_email(email):
                logger.error(f"Email {email} is not valid or accessible")
                return []
            
            # Get messages with retry
            for attempt in range(3):
                try:
                    response = self.session.get(
                        f"{self.base_url}",
                        params={
                            "action": "getMessages",
                            "login": login,
                            "domain": domain
                        }
                    )
                    response.raise_for_status()
                    messages_data = response.json()
                    
                    messages = []
                    for msg in messages_data:
                        try:
                            msg_id = msg.get('id')
                            if msg_id:
                                full_msg = self.get_message(email, str(msg_id))
                                if full_msg:
                                    messages.append(full_msg)
                        except Exception as e:
                            logger.error(f"Error processing message {msg.get('id')}: {str(e)}")
                            continue
                    
                    return messages
                    
                except Exception as e:
                    logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
                    if attempt < 2:  # Not the last attempt
                        sleep(2)  # Wait before retry
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting messages from 1secmail: {str(e)}")
            return []

    def get_message(self, email: str, message_id: str) -> Optional[Dict]:
        """Get a specific message"""
        try:
            login, domain = email.split('@')
            
            # Get message with retry
            for attempt in range(3):
                try:
                    response = self.session.get(
                        f"{self.base_url}",
                        params={
                            "action": "readMessage",
                            "login": login,
                            "domain": domain,
                            "id": message_id
                        }
                    )
                    response.raise_for_status()
                    msg = response.json()
                    
                    # Extract attachments if any
                    attachments = []
                    if msg.get('attachments'):
                        for attachment in msg['attachments']:
                            try:
                                att_response = self.session.get(
                                    f"{self.base_url}",
                                    params={
                                        "action": "download",
                                        "login": login,
                                        "domain": domain,
                                        "id": message_id,
                                        "file": attachment['filename']
                                    }
                                )
                                if att_response.status_code == 200:
                                    attachments.append({
                                        'filename': attachment['filename'],
                                        'content': att_response.content
                                    })
                            except Exception as e:
                                logger.error(f"Error downloading attachment: {str(e)}")
                    
                    return {
                        'id': str(msg.get('id')),
                        'from': msg.get('from'),
                        'subject': msg.get('subject'),
                        'date': msg.get('date'),
                        'body': msg.get('textBody', msg.get('htmlBody', '')),
                        'attachments': attachments
                    }
                    
                except Exception as e:
                    logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
                    if attempt < 2:  # Not the last attempt
                        sleep(2)  # Wait before retry
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting message from 1secmail: {str(e)}")
            return None

    def get_provider_name(self) -> str:
        """Get provider name"""
        return "1secmail"

    def get_available_domains(self) -> List[str]:
        """Get list of available domains"""
        return self.domains

class GuerrillaMailAPI(TempMailAPI):
    """Guerrilla Mail provider"""
    def __init__(self):
        """Initialize provider"""
        self.base_url = "https://api.guerrillamail.com/ajax.php"
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
            logger.info(f"Initialized Guerrilla Mail session: {self.session_id}")
        except Exception as e:
            logger.error(f"Error initializing session: {str(e)}")
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
            logger.info(f"Set email address: {email_user}@{domain}")
            return True
        except Exception as e:
            logger.error(f"Error setting email address: {str(e)}")
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
            logger.error(f"Error resetting session: {str(e)}")

    def generate_email(self) -> str:
        """Generate a new email address"""
        try:
            self._check_session()
            
            # Use more reliable pattern for username
            email_user = ''.join(random.choices(string.ascii_lowercase, k=4))  # Start with letters
            email_user += ''.join(random.choices(string.digits, k=4))  # Add some numbers
            email_user += ''.join(random.choices(string.ascii_lowercase, k=4))  # End with letters
            
            domain = "sharklasers.com"  # Use most reliable domain
            email = f"{email_user}@{domain}"
            
            if not self._set_email_address(email_user, domain):
                raise APIError("Failed to set email address")
            
            return email
        except Exception as e:
            logger.error(f"Error generating email: {str(e)}")
            raise APIError("Failed to generate Guerrilla Mail address")

    def get_messages(self, email: str) -> List[Dict]:
        """Get messages for an email address"""
        try:
            self._check_session()
            
            email_user = email.split('@')[0]
            domain = email.split('@')[1]
            
            # Set email address
            if not self._set_email_address(email_user, domain):
                return []
            
            # Get messages with retry
            for attempt in range(3):
                try:
                    response = self.session.get(
                        f"{self.base_url}",
                        params={
                            "f": "check_email",
                            "seq": "0",
                            "sid_token": self.session_id
                        }
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    messages = []
                    for msg in data.get('list', []):
                        messages.append({
                            'id': str(msg.get('mail_id')),
                            'from': msg.get('mail_from', 'Unknown'),
                            'subject': msg.get('mail_subject', 'No Subject'),
                            'date': msg.get('mail_timestamp', 'Unknown'),
                            'body': msg.get('mail_excerpt', '')
                        })
                    
                    if messages:
                        return messages
                    
                    # Wait before retry
                    sleep(2)
                    
                except Exception as e:
                    logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
                    if attempt == 2:  # Last attempt
                        self._forget_me()  # Reset session
                    else:
                        sleep(2)  # Wait before retry
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting messages: {str(e)}")
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
            
            return {
                'id': str(msg.get('mail_id')),
                'from': msg.get('mail_from', 'Unknown'),
                'subject': msg.get('mail_subject', 'No Subject'),
                'date': msg.get('mail_timestamp', 'Unknown'),
                'body': msg.get('mail_body', '')
            }
        except Exception as e:
            logger.error(f"Error getting message: {str(e)}")
            return None

    def get_provider_name(self) -> str:
        """Get provider name"""
        return "GuerrillaMailAPI"

    def get_available_domains(self) -> List[str]:
        """Get list of available domains"""
        return self.domains

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
