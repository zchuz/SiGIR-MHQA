This file is used to generate the critic dataset for our T2 model.

#### Relevance
Definition: Whether the retrieval coutent is relevant to the retrieval query.
Level:
- [Relevant] The retrieval content is irrelevant to the query.
- [Irrelevant] The retrieval content is relevant to the query.

#### Groundness
Definition: Whether the response is grounded on the retrieval content.
Level:
- [Fully supported] The response is fully based on the retrieval content.
- [Partially supported] The response is partially based on the retrieval content, with some imagination/hallucination.
- [Not supported] The response is not based on the retrieval content / The response does'nt provide any useful information.

#### Value
Definition: The value of the full reasoning process.
Level:
- [4] The reasoning process is clear and the answer is correct. (partial match = 1)
- [3] The reasoning process is clear and the answer is partially correct. (f1 > 0.7)
- [2] The reasoning process is flawed, with the answer partially correct. (f1 < 0.7)
- [1] The reasoning process is flawed, with the answer incorrect. (f1 = 0)
- [0] The reasoning doesn't provide the final answer. (answer = unknown)

-------------------------------------
| Prompts for relevance critic model|
-------------------------------------

You will be provided with a question, along with an evidence document.
Your job is to determine whether the evidence is relevant to the question, and provide explanation for your decision.
If the evidence is relevant to the question, you should response with [Relevant]; otherwise, response with [Irrelevant].

### Question: In which city did Shamir die?

### Evidence: Shamir (musician)
Shamir Bailey (born November 7, 1994), known mononymously as Shamir, is an American singer, songwriter and actor from Las Vegas, Nevada. His debut extended play (EP), "Northtown", was released in June 2014 to positive reviews. In October 2014 he signed to XL Recordings and released the single "On the Regular". His debut LP "Ratchet" was released on May 19, 2015 in the Unit

Rating: [Relevant]
Explanation: The evidence is relevant to the question because it identifies Shamir Bailey as a musician from Las Vegas.

### Question: What is the administrative territorial entity for Malta Township?

### Evidence: Minsk Region
Minsk Region or Minsk Voblasć or Minsk Oblast (, "Minskaja vobłasć" ; , "Minskaja oblastj") is one of the regions of Belarus. Its administrative center is Minsk, although it is a separate administrative territorial entity of Belarus. As of 2011, the region's population is 1,411,500.

Rating: [Irrelevant]
Explanation: The evidence is irrelevant to the question as it does not mention Malta Township in the document.

### Question: {}

### Evidence: {}

--------------------------------------
| Prompts for relevance critic model |
--------------------------------------

You will be provided with a question, an evidence document, along with a response.
Your job is to determine whether the response is supported by the evidence, and provide explanation for your decision.
Use the following scale to rate the response:
[Fully supported] Most of the information in the response is supported by the evidence.
[Partially supported] Some of the information in the response is supported by the evidence, but there are some parts that are speculative.
[Not supported] The response is not supported by the evidence / The response does not provide useful reasoning.

### Question: In which country is the Financial Conduct Authority located?

### Evidence: Martin Wheatley
Martin Wheatley is a British financier, formerly managing director of the Consumer and Markets Business Unit of the Financial Services Authority in the UK, and is the former CEO of the Financial Conduct.

### Response: From Document, we know that Martin Wheatley is a British financier who formerly managed the Consumer and Markets Business Unit of the Financial Services Authority in the UK. This indicates that the Financial Conduct Authority is located in the United Kingdom.

Rating: [Fully supported]
Explanation: The response is fully supported by the evidence, since it mentions Martin Wheatley's role in the UK's Financial Services Authority and his position as former CEO of the Financial Conduct Authority, to conclude that the FCA is located in the UK.

### Question: What is the birthplace of Jane Siberry?

### Evidence: Gavin Bradley 
Gavin Bradley is an award-winning Canadian singer-songwriter and producer based in Toronto who has worked with artists like Nelly Furtado, Tori Amos and Jane Siberry. Fusing acoustic and electronic elements, his work is identifiable for its signature "warm" piano sound and live strings mixed with filtered synthesizers and other electronic manipulations . Besides production, Bradley is a solo recording artist. His debut album 'Deep Freeze' was released on UMI Records in 2006.

### Response: The response is partially supported because the evidence mentions Gavin Bradley's Toronto connection but does not directly state Jane Siberry's birthplace. The inference about her possible connection to Toronto is speculative.

Rating: [Partially supported]
Explanation: The response is partially supported because the evidence mentions Gavin Bradley's Toronto connection but does not directly state Jane Siberry's birthplace. The inference about her possible connection to Toronto is speculative.

### Question: What is the city where WXPN is located?

### Evidence: Where is Mama
Where is Mama or Where's Mama (), is a short Chinese animated film produced by Shanghai Animation Film Studio in 1960 under the artistic guidance of Te Wei. The narrated film describes the adventures and misadventures of a group of tadpoles in search of their mother. It is one of Te Wei's first attempts to break away from Western style animation and aim for a painterly style influenced by Qi Baishi and more in keeping with native Chinese aesthetic sensibilities. Because of its simple story line and repetitive script it is ideal for children who are beginning their study of the Chinese language.

### Response: From Document, we know that "Where is Mama" is a short Chinese animated film produced by Shanghai Animation Film Studio in 1960. Since WXPN is a radio station, it is likely located in the city where the film was produced, which is Shanghai.

Rating: [Not supported]
Explanation: The response is not supported by the evidence. The evidence discusses a Chinese animated film produced by Shanghai Animation Film Studio, but it does not mention WXPN, which is a radio station. The response speculates that WXPN is likely located in Shanghai based on the film's production location, but this is not substantiated by the evidence provided.

### Question: {}

### Evidence: {}

### Response: {}

Rating:
