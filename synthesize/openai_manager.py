import openai
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import time
from typing import List, Dict
import threading
import fcntl


def repeat_until_calling_openai_api_successfully(func):
    def wrapper(*args, **kwargs):
        global account_manager
        while True:
            result = None
            account_manager = kwargs.get('account_manager', None)
            thread_id = kwargs.get('thread_id', None)
            account = account_manager.thread_to_account.get('thread_id', None)  # triad
            if account is None:
                account = account_manager.get_next_account(thread_id)
            openai.api_key = account[-1]
            try:
                result = func(*args, **kwargs)

            except openai.error.APIConnectionError as e:
                logger.info('openai connection error, so retry after sleeping 5 seconds')
                logger.info(e)
                time.sleep(5)

            except openai.error.RateLimitError as e:
                logger.info(type(e))
                logger.info(e)
                logger.info('e._message:{}'.format(e._message))
                if 'quota' in e._message:
                    logger.info('meet false quota error, so retry after sleeping 5 seconds')
                    time.sleep(5)
                else:
                    logger.info('openai rate limit error, so retry after sleeping 60 seconds')
                    logger.info(e)
                    time.sleep(60)
                    
            except openai.error.AuthenticationError as e:
                if 'This key is associated with a deactivated account' in e._message:
                    logger.info('the account {} is deactivated. so use next'.format(account[-1]))
                    logger.info(e)
                    account = account_manager.get_next_account(thread_id, account)
                    account_manager.thread_to_account[thread_id] = account
                else:
                    logger.info('meet unexpected AuthenticationError, so retry after sleeping 5 seconds')
                    logger.info(e)
                    account = account_manager.get_next_account(thread_id, account)
                    account_manager.thread_to_account[thread_id] = account

            except openai.error.OpenAIError as e:
                logger.info('meet unexpected openai error, so retry after sleeping 5 seconds')
                logger.info(e)
                time.sleep(5)

            except Exception as e:
                print('un encountered exception as follows:')
                print(e)
                time.sleep(5)

            if result != None:
                return result
            else:
                pass

    return wrapper


@repeat_until_calling_openai_api_successfully
def call_chatgpt(call_llm: callable, model: str, example: Dict, thread_id: int, account_manager: object) -> bool:
    messages = [{"role": "user", "content": example["prompt"]}]
    completion = call_llm(model=model, messages=messages, temperature=1.0, top_p=1.0, max_tokens=4096)
    ret = completion['choices'][0]['message']['content']
    example["example"] = ret
    return True


class OpenAI_Account_Manager_MultiThread_One_Acount_Many_Used:
    '''
    OpenAI_Account_Manager_MultiThread_One_Acount_Many_Used: when OpenAI_Account_Manager_MultiThread uses one account for one thread,
    so the number of threads is limited by the number of accounts.
    OpenAI_Account_Manager_MultiThread_One_Acount_Many_Used support multiple threads using one account.
    '''
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls)

        return cls._instance


    def __init__(self, used_account_fp: str, all_account_fp: str, limit_account_num: int=-1) -> None:
        """Class init
        Args
        ----
        used_account_fp: str
            Path to file containing used OpenAI accounts.
        all_account_fp: str
            Path to file containing all OpenAI accounts.
        limit_account_num: int=-1
            Number of available accounts.
        """
        if hasattr(self, 'inited'):
            return
        self.inited = 1
        self.now_account_idx = 0

        self.used_account_fp = used_account_fp
        self.all_account_fp = all_account_fp

        used_account_f = open(used_account_fp, 'r')
        used_account = list(map(lambda x: x.strip().split('----'), used_account_f.readlines()))
        used_account_f.close()

        all_account_f = open(all_account_fp, 'r')
        all_account = list(map(lambda x: x.strip().split('----'), all_account_f.readlines()))
        all_account_f.close()

        used_account_key = []
        for account in used_account:
            if len(account) == 4:
                used_account_key.append(account[-2])
            else:
                used_account_key.append(account[-1])

        all_account = list(filter(lambda x: x[-1] not in used_account_key, all_account))
        temp_all_account = []
        for account in all_account:
            if len(account) == 4 and account[-2] not in used_account_key:
                temp_all_account.append(account)
            elif len(account) == 3 and account[-1] not in used_account_key:
                temp_all_account.append(account)
            else:
                raise Exception
        all_account = temp_all_account

        if limit_account_num > 0:
            all_account = all_account[:limit_account_num]

        self.used_account = used_account
        self.used_account_key = set(used_account_key)
        self.all_account = all_account

        self.using_account = []
        self.thread_to_account = {}
        logger.info('successfully build OpenAI_Account_Manager, now the number of available accounts is {}'.format(len(self.all_account)))

        self.next_account_lock = threading.Lock()
        self.empty_account_lock = threading.Lock()


    def get_next_account(self, thread_id, last_empty_account=None):
        with self.next_account_lock:
            available_num = self.check_available_account_num()
            if available_num == 0:
                logger.info('all accounts used, so..')
            else:
                logger.info('now available accounts : {}'.format(available_num))

            while True:
                result = self.all_account[self.now_account_idx]
                if result[-1] in self.used_account_key or result[-2] in self.used_account_key:
                    self.now_account_idx += 1
                    self.now_account_idx = self.now_account_idx % len(self.all_account)
                else:
                    break

            result = self.all_account[self.now_account_idx]
            self.now_account_idx += 1
            self.now_account_idx = self.now_account_idx % len(self.all_account)

            if last_empty_account != None:
                self.record_empty_account(last_empty_account)
                logger.info('Thread {} account: [{}, {}, {}] '
                            'runs out'.format(thread_id,
                                              self.used_account[-1][0],
                                              self.used_account[-1][1],
                                              self.used_account[-1][2]))
                logger.info('Thread {} use next account: [{}, {}, {}] '
                            .format(thread_id, result[0],
                                    result[1],
                                    result[2]))
            else:
                logger.info('Thread {} first account: [{}, {}, {}] '
                            .format(thread_id, result[0],
                                    result[1],
                                    result[2]))
            return result


    def record_empty_account(self, empty_account):
        with self.empty_account_lock:
            self.used_account.append(empty_account)
            if len(empty_account) == 4:
                self.used_account_key.add(empty_account[-2])
            else:
                self.used_account_key.add(empty_account[-1])
            with open(self.used_account_fp, 'a') as tmp_used_account_f:
                fcntl.fcntl(tmp_used_account_f.fileno(), fcntl.LOCK_EX)
                print('----'.join(self.used_account[-1]), file=tmp_used_account_f)


    def check_available_account_num(self):
        available_num = 0
        for account in self.all_account:
            if len(account) == 4 and account[-2] not in self.used_account_key:
                available_num += 1
            elif len(account) == 3 and account[-1] not in self.used_account_key:
                available_num += 1
            else:
                raise Exception
        return available_num


def get_account_manager(
    account_file: str, 
    used_file: str, 
    multi_thread: bool=False, 
    limit_account_num: int=-1
) -> OpenAI_Account_Manager_MultiThread_One_Acount_Many_Used:
    """Get an instance of managing openai accounts.
    Args
    ----
    account_file: str
        The file containing available username, password and key of OpenAI API account.
    used_file: str
        The file containing unavailable username, password and key of OpenAI API account.
    multi_thread: bool=False
        Whether to use multi-thread or not.
    limit_account_num: int=-1
        Number of available accounts.

    Returns
    -------
    account_manager: OpenAI_Account_Manager_MultiThread_One_Acount_Many_Used
        An instance of class OpenAI_Account_Manager_MultiThread_One_Acount_Many_Used
    """
    if multi_thread:
        account_manager = OpenAI_Account_Manager_MultiThread_One_Acount_Many_Used(used_file, account_file, limit_account_num=limit_account_num)
    else:
        raise NotImplementedError()
    return account_manager
