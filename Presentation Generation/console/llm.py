import openai, os, json, base64, io, numpy as np, cv2
from dotenv import find_dotenv, load_dotenv
# from langchain.chat_models import ChatOpenAI #chat AI (turning deprecated)
from langchain_openai import ChatOpenAI #chat AI
from langchain.prompts import ChatPromptTemplate #prompt
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser #output to json
from langchain.callbacks import get_openai_callback #check tokens
# from langchain.utils.openai_functions import convert_pydantic_to_openai_function 
from langchain_core.utils.function_calling import convert_to_openai_function
from typing import List,Optional
# from pydantic import BaseModel, Field #container of functions, incompatible
from langchain_core.pydantic_v1 import BaseModel, Field
from pprint import pprint as pp

import requests, matplotlib.pyplot as plt, random, ppt as Prs, tqdm
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup
from PIL import Image


'''
General settings
change the system_message for use
'''
config = {'batch':{"max_concurrency": 30},
          'search':{'engine':['https://www.google.com/','https://www.baidu.com/']},
          'system_message' : '''You are a helpful assistant.''',
          'Presentation':{'min_slides':15,'word':50},
          'file_trasnfer': 'scp "liborui@node-q:/local_home/liborui/stable_diffusion/generative-models/output/output.png" C:/Users/Borui/Dean/Computing/AI/Others/UROP/PPT_Generation/output/',
}
load_dotenv(find_dotenv())
openai.api_key=os.environ['OPENAI_API_KEY']
system_message = 'You are a helpful assistant.'
get_prompt=lambda x,system_message='': [{"role":"system","content":system_message if system_message=='' else system_message},{"role":"user","content":x}]
get_lc_prompt = lambda x='':ChatPromptTemplate.from_messages([("system", system_message if x=='' else x),("user", "{input}")])
quiet=False


'''
'''


'''functions'''
def callback(f):
    def wrapper(*params,**kwparams):
        try:
            if quiet == False:
                with get_openai_callback() as cb:
                    res=f(*params,**kwparams)
                    print(cb)
            else:
                res=f(*params,**kwparams)
            return res
        except Exception as e:
            print(e)
            return None
    return wrapper
def create_chain(model_name="gpt-3.5-turbo",system_message='',temperature=0.3,functions_config={'functions':None,'function_call':{"name": 'Name of the function'}},**params):
    '''gpt-3.5-turbo-16k;gpt-4-1106-preview;gpt-4-turbo-preview
    use chain to invoke/batch response
    system_message: system message'''
    model = ChatOpenAI(temperature=temperature, model=model_name)
    if functions_config['functions']:
        if type(functions_config['functions'])!=dict:
            functions_config['functions']=[convert_to_openai_function(f) for f in functions_config['functions']] if type(functions_config['functions'])==list else [convert_to_openai_function(functions_config['functions'])]
        model=model.bind(**functions_config)
        chain = {"input": lambda x: x} | get_lc_prompt(system_message) | model | JsonOutputFunctionsParser()
    else:
        chain = {"input": lambda x: x} | get_lc_prompt(system_message) | model
    return chain 
@callback
def chat(user_message,chain=create_chain(),force_invoke=False):
    resp=chain.batch(user_message,**config["batch"]) if type(user_message)==list and not force_invoke else chain.invoke(user_message)
    return resp
def vision(image_path="../image/Undergraduate-Research-Opportunities-Programme_scaled1.jpg",user_message="Analyze the image.",type='png',detail='low'):
    '''detail: low, high, or auto \ntype:.format'''
    # OpenAI API Key
    api_key = openai.api_key
    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    # Getting the base64 string
    base64_image = encode_image(image_path)
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }
    payload = {
    "model": "gpt-4-vision-preview",
    "messages": [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": user_message},
            {"type": "image_url",
            "image_url": {
                "url": f"data:image/{type};base64,{base64_image}",
                "detail": detail}
            }
        ]
    }],
    "max_tokens": 300
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload,timeout=None)
    # convert bin to str
    json_string = response.content.decode('utf-8')
    json_data = json.loads(json_string)
    return json_data
def search(query='', engine='https://www.google.com/search?q=', mode='text'):
    '''Perform a web search and parse titles and URLs from the search results.
    mode: text/image/icon
    return list: [{'title':title,'url':url},]'''
    res=[];engine= 'https://www.flaticon.com/search?' if mode=='icon' else engine
    params = {'image':{'q': query,'tbm': 'isch' if mode == 'image' else '',},
              'text':{'q': query},
              'icon':{'type':'icon','word':query},}
    resp = requests.get(engine, params=params[mode])
    soup = BeautifulSoup(resp.content, 'html.parser')# Use BeautifulSoup to parse the HTML content
    if mode=='text':
        for h3 in soup.find_all('h3'):# Google uses <h3> tags for titles within <a> tags for links in its search results
            a = h3.find_parent('a')
            if a:
                title = h3.get_text()
                link = a['href']# think is the raw link
                parsed_url = urlparse(link)# Google search results contain URLs in a /url?q= format
                query_string = parse_qs(parsed_url.query)
                actual_url = query_string.get('q', [None])[0]
                if actual_url:
                    res.append({'title':title,'url':actual_url})
        return res
    elif mode=='image':
        # Parsing for image mode (this is highly subject to change)
        images = soup.find_all('img', attrs={'class': 't0fcAb'})
        for img in images:
            src = img['src'] if 'src' in img.attrs else img['data-src']
            alt = img['alt']
            res.append({'title': alt, 'url': src})
        return res
    elif mode=='icon':
        images=soup.find_all('a',attrs={'class':"view link-icon-detail"}) 
        for cnt,img in enumerate(images):
            res.append(img.contents[1]['src'])
            if cnt==10:  break
        return res
def get_url_content(url,mode='text',context_window=250,stream=False):
    '''context_window: max len of str for classification
    mode: text/image/icon
    return: text:{'soup':,'title':,'content':}; icon: '''
    res = requests.get(url)
    if mode=='text':
        soup = BeautifulSoup(res.text, 'html.parser')
        content=[None]*2 # 0:article, 1:p
        title = soup.find('h1', class_='fs-headline') # title
        if title:
            title=title.text;content[0] = soup.find('div', class_='fs-article').text # To extract the article content
            res=content[0]
        else:
            p=soup.find_all('p')
            if len(p)<10:
                content[1]=list(map(lambda x: x.text,p)) # certain websites
                test=list(map(lambda x: x if len(x)<context_window else x[:context_window],content[1][:3]))
            else:
                content[1]=soup.text; length = len(content[1])
                test=[content[1][i:i+context_window] for i in range(0,length-context_window,(length-context_window)//3)] if length>context_window else [content[1]]
            resp=chat(test,create_chain(temperature=0,system_message=f'is it useful info related to {soup.title.text}? respond with only y or n'))
            res=content[1] if 'y' in [i.content for i in resp] else ''
        return {'soup':soup,'title':soup.title.text,'content':res}
    elif mode in ['image','icon']:
        resp=requests.get(url)
        im_stream=io.BytesIO(resp.content);im_stream.seek(0)
        image_array = np.frombuffer(resp.content, dtype=np.uint8)
        res = cv2.cvtColor(cv2.imdecode(image_array, cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        return im_stream if stream else res
def get_ppt(name='../output/ppt/test.pptx',icon_width=1+1/3,process=False,title=None,text=None,vis=None,topic=None,slides=config['Presentation']['min_slides'],skip_vis=False,chain=None):
    '''icon_width: roughly the width for icons
    transparent bg can only be .png'''
    if process or not vis:
        topic=topic if topic else 'GPT' # content
        resp=Text.process(topic,slides=slides,chain=chain)
        title,text,vis=resp.values();Ideation.set_element(*resp.values())
        vis=[i['visual'] for i in vis]
    else:
        if not vis:
            vis=[['GPT model', 'Deep learning'], ['AI model', 'text generation'], ['large dataset', 'predict the next word in a sentence'], ['Natural Language Processing (NLP) model', 'Uses deep learning to generate human-like text', 'Contextual understanding of text'], ['GPT-3', 'AI assistants'], ['human-like text', 'time and effort', 'personalized responses', 'customer service interactions'], ['Lack of common sense', 'Bias'], ['GPT-2', 'GPT-3'], ['transformer architecture', 'text generation'], ['AI model', 'chatbots'], ['GPT model'], ['GPT generating text', 'Authors using GPT for writing'], ['GPT model', 'Code generation'], ['potential biases in data', 'responsibility of developers', 'impact of AI-generated content on society'], ['GPT model', 'Healthcare', 'Finance', 'Entertainment']]
    ppt=Prs.PPT() # ppt generation
    if title and text:
        for i in range(len(text)):
            slide=ppt.add_slide(slide_layout=5) # only title
            ppt.add_img(slide,img='../output/bg_transparent.png',pos=[0,0],width=ppt.prs.slide_width,height=ppt.prs.slide_height)
            ppt.add_text_chunk(slide.shapes,list(map(lambda x:Prs.Emu(x),[457200, 1600200, 8229600*0.8, 4525963])))
            ppt.change_text(slide.shapes[0],title[i])
            ppt.change_text(slide.shapes[2],text[i])
            ppt.set_font(slide.shapes[2],size=18)
    
    if not skip_vis:
        for slide,imgs in zip(ppt.prs.slides,vis):
            _=len(imgs);step=(7.5-_*icon_width)/(_+1);cur_h=step
            for img in imgs:
                # try:
                    resp=search(img,mode='icon')
                    if len(resp)==0:
                        continue
                    img=random.choice(resp[:min(len(resp),10)])
                    stream=get_url_content(img,mode='icon',stream=1)
                    h,w=[i/96 for i in plt.imread(stream).shape[:2]]
                    stream.seek(0);ppt.add_img(slide,[min(9,10-w),cur_h],stream,icon_width)
                    # shape=slide.shapes[1]
                    # ppt.set_pos(shape,[shape.left,shape.top,shape.width*0.8,shape.height]) # problematic
                    cur_h+=h+step
                # except Exception as e:
                #     print(e)
    ppt.prs_save(name=name)
class Ideation:
    title,text,vis,ideas=[None]*4
    def set_element(title=None,text=None,vis=None):
        Ideation.title,Ideation.text,Ideation.vis=title,text,vis
    def ideate(prompt='',system_message='You are a helpful and creative assistant, ideate and provied facts, real-life scenarios & entities as the input instructs. Respond with only short & succinct points',temperature=0.3,**params):
        '''prompt: the topic or core info'''
        queries=['Relevant Timelines(When): Identifying when important events happened or when key figures made significant contributions. This includes dates, historical periods, or important phases.',
        'Associated Places(Where): Finding the places or areas where events happened or where important people were active. This ranges from specific sites to wider regions.',
        'Key Individuals or Groups(Where): Focusing on the main people or groups involved in or affected by the events or topics. This includes both major and minor figures.',
        'Underlying Reasons(Why): Looking into why events happened or why people took certain actions. This is about understanding the motivations or causes.',
        'Methods and Processes(How): Explaining how things were done or achieved, including the techniques, strategies, or methods used.',
        'Core Subject Matter(What): Summarizing the main events, discoveries, contributions or influence related to the topic, focusing on the essential aspects or outcomes.']
        queries=list(map(lambda x:''.join([f'Topic: {prompt}\n',x]),queries))
        return chat(queries,create_chain(temperature=temperature,system_message=system_message))
    def converge(llm=[],scrapping={},system_message='You are a helpful and creative assistant, only summarize the main & important & authentic points and offer fun facts if possible.',temperature=0,**params):
        '''converge the llm and scrapping info
        llm: the res from chat 
        scrapping: the res from scrapping;{'content':obj,...}'''
        res=[]
        if len(llm)>0 and type(llm[0])==str:
            res=llm.split('\n')
        else:
            for i in llm:
                res.extend(i.content.split('\n'))
        if len(scrapping)>1:
            res.extend(scrapping['content'])
        return chat(res,create_chain(system_message=system_message,temperature=temperature))
class Text:
    resp=None
    def process(content=[],slides=config['Presentation']['min_slides'],chain=None):
        chain=chain if chain else create_chain(functions_config={ 'functions': Outline,'function_call': { "name": 'Outline' }},system_message=f'''you are a helpful assistant to create a ppt. There should be at least {slides} body slides''')
        resp=chat(content,chain)
        title=resp['slides']
        resp1=chat([f'''topic:{content};title:'''+i for i in title],force_invoke=0,chain=create_chain(functions_config={ 'functions': Slide,'function_call': { "name": 'Slide' }},temperature=0,system_message=f'''Fill in the content according to the guide. it must contain illustrations by means of specific examples (who, when, where) and must offer insights into the topic. you must use layman\'s terms to explain complex concepts. you must engage the audience by posing questions or present hypothetical scenarios. make sure the length is no more than {config["Presentation"]['word']} words for one slide''',))
        txt=[i['content'] for i in resp1]
        vis=chat(txt,force_invoke=0,chain=create_chain(functions_config={ 'functions': Visual,'function_call': { "name": 'Visual' }},temperature=0,))
        return {'title':title,'txt':txt,'vis':vis}
    def get_presentation_text(resp=[]): # obsolete
        '''obsolete
        resp: result of chat(Presentation)
        return: list of text for each slide'''
        text=[resp['introduction'][0]['content']]
        text.extend([i['content'] for i in resp['body']])
        text.extend([resp['conclusion'][0]['content']])
        return text
    def increase_slides(resp=[],topic='',ideas='',slides=config['Presentation']['min_slides']):
        '''resp: list; titles from chat(Outline)
        return: list; titles with increased amount'''
        title=chat(f'''topic:{topic};ideas:{ideas};original slides:{";".join(resp[1:-1])}''',chain=create_chain(system_message=f'''Increase the no. of original ppt slides based on info. A slide is denoted by its title, eg. 'Introduction','Conclusion', etc. output the increased slide titles in the correct sequence starting with intro and ending with conclusion. Output format:"title1\ntitle2... There should be more than {slides} slides"'''))
        title=title.content.split('\n')
        return title
    def pdf(input_pdf_path, num_pages=3):
        import fitz
        text=f'document: {input_pdf_path.split("/")[-1]}\n'
        pdf_document = fitz.open(input_pdf_path)
        for page_number in range(min(num_pages, pdf_document.page_count)):
            page = pdf_document.load_page(page_number)
            page_text = page.get_text("text")
            text += page_text
        pdf_document.close()
        return text
class Visual:
    def get_transparent(img='../output/bg.png',transparency=0.4):
        im=cv2.imread(img)
        im=cv2.cvtColor(im,cv2.COLOR_BGR2BGRA)
        im[:,:,3]=255*transparency
        cv2.imwrite('../output/bg_transparent.png',im)
        return im
    def get_im_from_server(prompt="high-res, 8k, photorealistic, beautiful, ultra high res, beautiful scenery in Japan, mountains"):
        os.system(f'''ssh node-q source /local_home/envs/miniconda3/etc/profile.d/conda.sh; conda activate base; python3 /local_home/liborui/stable_diffusion/generative-models/src/diffuser_generator.py --prompt {prompt}; exit''')
        os.system('scp liborui@node-q:/local_home/liborui/stable_diffusion/generative-models/output/bg.png C:/Users/Borui/Dean/Computing/AI/Others/UROP/PPT_Generation/output/')
        return Image.open('../output/bg.png')
class Outline(BaseModel):
    f'''the blueprint and guideline for ppt creation. There should be at least {config['Presentation']['min_slides']} slides'''
    slides: List[str]=Field(description='''The main content and ideas for each slide.''')
class Slide(BaseModel):
    """the textual content of a ppt slide."""
    content: str=Field(description=f'''the specific text content of the slide. point form, succinct text, should dissect the topic instead of general statements. Explain important info and concepts via deconstruction and structuralism if never mentioned, it must give illustrations (examples or relevant incidents with who, when, where, why, how, what) and insights. It must use layman\'s terms to explain complex concepts. It must engage the audience: It must pose questions or present hypothetical scenarios. Each slide has a word limit of {config["Presentation"]['word']}''')
class Breakdown(BaseModel):
    f'''In the process of PPT creation and layout generation, all the texts are initially clumped together. Divide it into smaller sections with appropriate amount of text (breakdown) to be put at different locations. Given the text for the slide, output the divided sections.'''
    breakdown: List[str]=Field(description='''the text section breakdown''')
class Visual(BaseModel):
    visual: List[str]=Field(description='''Visual/image for each slide is crucial and imperative. Follow the step: 1.Find the core contents of the slide 2.Pick only 1 or 2 most important parts that will need an image/icon for illusrtation. The output is the obj or its description, less than 3 instances, not a link''')

