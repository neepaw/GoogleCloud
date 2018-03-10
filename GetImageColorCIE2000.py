
#######################################################################
# Copyright (C)                                                       #
# 2017 - 2018 IntelloCart (intellocart.com)                           #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################





# Install ColorMath, PIL, Numpy, 


# In[13]:


# Standard Color and RGB Code
standard_color_names = [ 'black', 'brown', 'grey', 'navy blue',
                        
                        'blue', 'tan', 'white', 'green', 'red', 
                        
                        'olive', 'beige', 'maroon',
                        
                        'orange', 'yellow', 'pink', 'off white',
                        
                        'teal', 'khaki', 'fluorscent green', 'purple',
                        
                        'mustard', 'silver', 'cream', 'coral',
                        
                        'gold', 'rust', 'lime green', 'bronze',
                        
                        'peach', 'sea green', 'lavender', 'turquoise', 
                        
                        'mushroom brown', 'gunmetal', 'metallic', 'magenta',
                        
                        'copper' ]
    
    
    
standard_color_codes = [ (0,0,0), (142,61,32), (147,159,163), (60,68,119),
                        
                        (0,92,221), (209,173,121), (255,255,255),(38,176,74), (224,40,61), 
                        
                        (0,148,96), (229,229,199), (183,3,78),
                        
                        (255,126,0), (230,222,0), (253,152,188), (240,240,240), 
                        
                        (0,118,117), (191,168,128), (110,194,37), (129,0,118), 
                        
                        (205,148,0), (171,171,171), (235,230,172), (255,105,38),
                        
                        (229,197,15), (189,37,0), (29,183,56), (209,115,23),
                        
                        (255,228,164), (0,132,69), (210,208,228), (0,227,206),
                        
                        (185,131,77) , (212,209,177), (224,202,188), (192,44,151),
                        
                        (169,92,24) ]


# In[14]:


import numpy as np
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colormath.color_diff import delta_e_cie1994
from colormath.color_diff import delta_e_cmc

import os
from PIL import Image
    
# Load and save the image
class Image_Color():
    
    def __init__(self, standard_color_codes, standard_color_names, background_replace_color):
        self.standard_color_codes = standard_color_codes
        self.standard_color_names = standard_color_names
        self.img = None
        self.background_color_index = None
        self.filename = ""
        self.same_color_category = {}
        self.background_replace_color = background_replace_color
        self.colors = ""
    
#     def import_all_libraries(self):
#         import numpy as np
#         from colormath.color_objects import sRGBColor, LabColor
#         from colormath.color_conversions import convert_color
#         from colormath.color_diff import delta_e_cie2000
        
        
    def download_image(self,url,filename):
        from urllib import request
        try:
            request.urlretrieve(url, filename)
            return True
        except ValueError:
            return False
            
    
    def load_image(self,filename):
        self.img = Image.open(filename)
    
    
    
    def get_colors_frequency(self, n):
        im_rgb = self.img.convert('RGB')
        colors = im_rgb.getcolors(maxcolors = 100000)
        colors.sort()
        n = 0 - n
        top_colors = colors[:n :-1]
        return top_colors
    
    
    def is_similar(self,color1,color2):
        r1 = color1[0]
        g1 = color1[1]
        b1 = color1[2]
        r2 = color2[0]
        g2 = color2[1]
        b2 = color2[2]
        color1_rgb = sRGBColor(r1,g1,b1, is_upscaled = True)
        color2_rgb = sRGBColor(r2,g2,b2, is_upscaled = True)
        color1_lab = convert_color(color1_rgb, LabColor)
        color2_lab = convert_color(color2_rgb, LabColor)
        delta_e = delta_e_cie2000(color1_lab, color2_lab)
        return delta_e
    
    
    def get_important_color(self):
        import numpy as np
        self.initialize_same_color_category_dict()
        relevant_colors_original, colors_name_original ,background_color_index = self.get_relevant_color()
        self.change_background(background_color_index)
        relevant_colors_changed, colors_name_changed ,_ = self.get_relevant_color()
        colors_final = self.get_common_colors(colors_name_changed, colors_name_original, relevant_colors_original)
        # print(colors_final)
        return colors_final
    
    
    
    def get_common_colors(self, colors_name_changed, colors_name_original, relevant_colors_original):
        common_colors = [(color,percentage) for color,percentage in relevant_colors_original if color in colors_name_changed and (color in colors_name_original)]
        return common_colors
        
        
    def change_background(self,background_color_index):
        im = self.img
        im = im.convert('RGBA')
        data = np.array(im)
        rgb = data[:,:,:3]
        background_colors = np.array(self.same_color_category[background_color_index])
        for bgc in background_colors:
            mask = np.all(rgb == bgc, axis = -1)
            data[mask] = self.background_replace_color
        self.img = Image.fromarray(data)
    
    
    def initialize_same_color_category_dict(self):
        for i in range(0,len(standard_color_codes)):
            self.same_color_category[i] = []
    
    
    def get_relevant_color(self):
        import numpy as np
        freq_colors = self.get_colors_frequency(2000)
        overall_similarity = np.zeros(len(self.standard_color_codes))
        total_imp_color_pixel = 0
        for color in freq_colors:
            close_color_index = self.closest_color(color)
            self.same_color_category[close_color_index].append(np.array(color[1]))
            overall_similarity[close_color_index] = overall_similarity[close_color_index] + color[0]
            total_imp_color_pixel = total_imp_color_pixel + color[0]
        overall_similarity = overall_similarity * 100/ (total_imp_color_pixel - freq_colors[0][0])
        relevant_colors_index = overall_similarity.argsort()[:-10 :-1]
        relevant_colors = []
        colors_name = []
        for index in relevant_colors_index:
            color_name = self.standard_color_names[index]
            color_percentage = overall_similarity[index]
            val = (color_name,color_percentage)
            relevant_colors.append(val)
            colors_name.append(color_name)
        return relevant_colors[1:], colors_name ,relevant_colors_index[0]
        

        
    def closest_color(self,color):
        min_similarity = 100000
        index = 0
        color = np.array(color[1])
        for i in range(0,len(self.standard_color_codes)):
            curr_similarity = self.is_similar( color , np.array(self.standard_color_codes[i]) )
            if curr_similarity < min_similarity:
                min_similarity = curr_similarity
                index = i
        return index


# In[15]:


def get_rgb(color):
    rgb = ''
    for i in range(len(standard_color_names)):
        if standard_color_names[i] == color:
            return np.array(standard_color_codes[i])
    return []

def get_color(url,filename,background_color):
    img_col = Image_Color(standard_color_codes, standard_color_names, background_color)
    is_url_correct = img_col.download_image(url,filename)
    if is_url_correct:
        img_col.load_image(filename)
        colors = img_col.get_important_color()
        return is_url_correct,colors
    else:
        return is_url_correct,None


# In[16]:


# Testing
def ensemble_multi_color(urls):
    # print('started')
    all_colors = {}
    for url in urls:
        filename = 'curr_image.jpg'
        background_color = [0,0,0,255]
        is_url_correct,colors = get_color(url,filename,background_color)
        if is_url_correct:
            for i in range(0,3):
                if colors[i][0] in all_colors:
                    all_colors[colors[i][0]] = all_colors[colors[i][0]] + colors[i][1]
                else:
                    all_colors[colors[i][0]] = colors[i][1]
    colors = sorted(all_colors.items(), key=lambda x:x[1])
    # print(colors)
    output = {}
    pref = ['third','second','first']
    val = ['color','RGB','percent']
    if len(colors) > 0:
        for i in range(0,3):
            small_output = {}
            small_output['color'] = colors[i][0]
            small_output['RGB'] = get_rgb(colors[i][0])
            small_output['percent'] = colors[i][1]
            output[pref[i]] = small_output
        os.remove(filename)
    else:
        for i in range(0,3):
            small_output = {}
            small_output['color'] = 'None'
            small_output['RGB'] = 'None'
            small_output['percent'] = 'None'
            output[pref[i]] = small_output
    return output
    


# In[17]:


def ensemble_color(url):
    # print('started')
    filename = 'curr_image.jpg'
    background_color = [0,0,0,255]
    is_url_correct,colors = get_color(url,filename,background_color)
    output = {}
    pref = ['first','second','third']
    val = ['color','RGB','percent']
    if is_url_correct:
        for i in range(0,3):
            small_output = {}
            small_output['color'] = colors[i][0]
            small_output['RGB'] = get_rgb(colors[i][0])
            small_output['percent'] = colors[i][1]
            output[pref[i]] = small_output
    else:
        for i in range(0,3):
            small_output = {}
            small_output['color'] = 'None'
            small_output['RGB'] = 'None'
            small_output['percent'] = 'None'
            output[pref[i]] = small_output
    os.remove(filename)
    return output


# In[18]:


# def print_colors(colors):
#     print('\n','\n')
#     print("Important Colors: ")
#     for color in colors:
#         print(color)


# In[19]:


#urls = [ 'https://assets.myntassets.com/h_640,q_90,w_480/v1/assets/images/2155365/2017/10/3/11507017293294-Red-Tape-Athleisure-Sports-Range-Men-Blue-Running-Shoes-6871507017293130-1.jpg',
#         'https://assets.myntassets.com/h_640,q_90,w_480/v1/assets/images/2155365/2017/10/3/11507017293272-Red-Tape-Athleisure-Sports-Range-Men-Blue-Running-Shoes-6871507017293130-2.jpg',
#         'https://assets.myntassets.com/h_640,q_90,w_480/v1/assets/images/2155365/2017/10/3/11507017293252-Red-Tape-Athleisure-Sports-Range-Men-Blue-Running-Shoes-6871507017293130-3.jpg',
#         'https://assets.myntassets.com/h_640,q_90,w_480/v1/assets/images/2155365/2017/10/3/11507017293229-Red-Tape-Athleisure-Sports-Range-Men-Blue-Running-Shoes-6871507017293130-4.jpg',
#           'https://assets.myntassets.com/h_640,q_90,w_480/v1/assets/images/2155365/2017/10/3/11507017293205-Red-Tape-Athleisure-Sports-Range-Men-Blue-Running-Shoes-6871507017293130-5.jpg' 
#]



#urls = [ 
        #'rukminim1.flixcart.com/image/832/832/shoe/p/v/2/visgre-utiblu-shosli-solonyx-1-0-m-adidas-8-original-imaezftxyzfh7u7h.jpeg?q=70'
 #        'https://rukminim1.flixcart.com/image/832/832/shoe/p/v/2/visgre-utiblu-shosli-solonyx-1-0-m-adidas-8-original-imaezftx4gpprrbz.jpeg?q=70'
#         'https://rukminim1.flixcart.com/image/832/832/shoe/p/v/2/visgre-utiblu-shosli-solonyx-1-0-m-adidas-8-original-imaezftxsjhve7un.jpeg?q=70',
#        'https://rukminim1.flixcart.com/image/832/832/shoe/p/v/2/visgre-utiblu-shosli-solonyx-1-0-m-adidas-8-original-imaezftxk7qte5yf.jpeg?q=70'
 #]


# urls = [
#     'https://assets.myntassets.com/h_640,q_90,w_480/v1/assets/images/1341633/2016/5/20/11463719982258-Roadster-Men-Casual-Shoes-7381463719982078-1.jpg',
#     'https://assets.myntassets.com/h_640,q_90,w_480/v1/assets/images/1341633/2016/5/20/11463719982244-Roadster-Men-Casual-Shoes-7381463719982078-2.jpg',
#     'https://assets.myntassets.com/h_640,q_90,w_480/v1/assets/images/1341633/2016/5/20/11463719982229-Roadster-Men-Casual-Shoes-7381463719982078-3.jpg',
#     'https://assets.myntassets.com/h_640,q_90,w_480/v1/assets/images/1341633/2016/5/20/11463719982217-Roadster-Men-Casual-Shoes-7381463719982078-4.jpg',
#      'https://assets.myntassets.com/h_640,q_90,w_480/v1/assets/images/1341633/2016/5/20/11463719982205-Roadster-Men-Casual-Shoes-7381463719982078-5.jpg'
# ]
# urls = ['https://assets.myntassets.com/h_640,q_90,w_480/v1/assets/images/2409848/2018/2/1/11517465404359-Men-Adidas-Sports-Shoes-CAFLAIRE-5101517465404251-1.jpg']
# urls = ['https://assets.myntassets.com/h_640,q_90,w_480/v1/assets/images/2409848/2018/2/1/11517465404288-Men-Adidas-Sports-Shoes-CAFLAIRE-5101517465404251-5.jpg']
# urls = ['https://assets.myntassets.com/h_640,q_90,w_480/v1/assets/images/2375708/2018/1/18/11516278954393-Nike-Men-Green-Skate-Shoes-5771516278954318-6.jpg']


# In[20]:


def node_to_python():
    urls = input()
    if urls[0] != '[':
        return ensemble_color(urls)
    else:
        urls = urls[1:].split('::')
        # print(urls)
        return ensemble_multi_color(urls)


# In[21]:


# Format for images color detection
## For single image => Pass URL without quotes
## For multiple images => [URL1::URL2::URL3:: and so on
#   (put opening square bracket in starting only and don't use quotes for urls. Separate all urls by '::' )


# In[25]:


colours = node_to_python()
print(colours)

