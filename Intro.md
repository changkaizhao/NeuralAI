# **NeuroAI 领域经典、前沿文献推荐清单**

**2022-11-16**

本路径聚焦视觉、语言和学习领域神经科学与人工智能的相关研究，邀请北京师范大学柳昀哲、北京大学鲍平磊和昌平实验室吕柄江三位研究员对相关领域文献进行深入梳理，文末提供文献PDF下载链接。

## **NeuroAI背景介绍**
---
从历史上看，神经科学一直是人工智能进步的关键驱动力和灵感来源，尤其是让人工智能更精通的领域，如视觉、基于奖励的学习、与物理世界的互动以及语言方面等，这些领域人类和动物也很擅长。当然，人工智能也有效推动着神经科学的发展，帮助研究人员打破原有的以理论和假设作为主导的研究范式，用模型和数据去构建从神经元到全脑的模型，从而实现模拟和预测等功能。

NeuroAI作为神经科学和人工智能的交叉点，其新兴领域基于这样一个前提：对神经计算的更好理解将揭示智能的基本成分，并催化人工智能的下一次革命，最终实现人工智能的能力与人类相匹配，甚至可能超过人类。


## **视觉智能**
---
推荐人：鲍平磊
2003年于中国科学技术大学获得理学学士学位，2014年于南加州大学获得博士学位，2014年-2020年在加州理工学院从事博士后研究。2020年11月至今任北京大学心理与认知科学学院研究员、麦戈文脑研究所研究员、北京大学-清华大学生命科学联合中心研究员。
鲍平磊实验室致力于高级视觉功能认知的神经机制探索，采用电生理，脑功能成像，微电刺激以及心理物理的方法等多种研究手段对于人和非人灵长类的视觉系统进行多层面的研究，并结合深度学习网络等多种手段去构建视知觉的数学模型。

这次以深度神经网络为标志的人工智能的浪潮起源于计算机视觉任务的探究，而人工智能的发展也同样反哺了视觉神经科学的发展。许多的研究发现利用深度神经网络提供了目前最好的模型来解释视觉系统不同层级对于不同刺激的反应，但也有研究指出生物视觉系统与人工智能之间存在着本质的差异，更也有批评者认为用机制尚且不清楚地深度神经网络来理解神经系统中的视觉系统，本质上就是利用一个黑箱去代替另外一个黑箱。但无论怎么说，二者之间的相互发展，相互启迪是学科交叉的一个典范。在这一模块的分享中，主要从视觉系统能否帮助我们设计出鲁棒性更高的神经网络算法和神经网络算法能否帮我们理解视觉回路两个方面去探究视觉信息的编码机制。

这篇文献利用深度学习网络来研究视觉系统的文章，是这一领域的开山之作。

>[Performance-optimized hierarchical models predict neural responses in higher visual cortexYamins Daniel L.K.（2014）](./%E8%A7%86%E8%A7%89%E6%99%BA%E8%83%BD/1-Performance-optimized%20hierarchical%20models%20predict%20neural%20responses%20in%20higher%20visual%20cortex.pdf)

这篇文献是发起人老师鲍平磊2020年的一项研究，利用深度网络对物体识别区域功能组织原则的探索。

>[A map of object space in primate inferotemporal cortexBao, Pinglei,She.et al.nature（2020）](./%E8%A7%86%E8%A7%89%E6%99%BA%E8%83%BD/2-A%20map%20of%20object%20space%20in%20primate%20inferotemporal%20cortex.pdf)

这篇文献研究说明无监督的深度神经网络也可以用来对腹侧视觉通路建模。

>[Unsupervised neural network models of the ventral visual streamChengxu Zhuang,Siming Yan,Aran Nayebi.et al.proceedings of the national academy of sciences(2021)](./%E8%A7%86%E8%A7%89%E6%99%BA%E8%83%BD/3-Unsupervised%20neural%20network%20models%20of%20the%20ventral%20visual%20stream.pdf)


这篇文献研究发现前馈网络可能并不足够描述腹侧通路，可能还需要循环神经网络。
>[Evidence that recurrent circuits are critical to the ventral stream’s execution of core object recognition behaviorKar, Kohitij,Kubilius.et al.nature neuroscience（2019）](./%E8%A7%86%E8%A7%89%E6%99%BA%E8%83%BD/4-Evidence%20that%20recurrent%20circuits%20are%20critical%20to%20the%20ventral%20stream%E2%80%99s%20execution%20of%20core%20object%20recognition%20behavior..pdf)



这篇文献使用深度生成网络进化的神经元图像揭示了视觉编码原理和神经元偏好。

>[Evolving Images for Visual Neurons Using a Deep Generative Network Reveals Coding Principles and Neuronal PreferencesCarlos R. Poncecell（2019）](./%E8%A7%86%E8%A7%89%E6%99%BA%E8%83%BD/5-Evolving%20images%20for%20visual%20neurons%20using%20a%20deep%20generative%20network%20reveals%20coding%20principles%20and%20neuronal%20preferencespdf.pdf)


这篇文献是关于利用神经网络寻找神经元的最优刺激。

>[Neural population control via deep image synthesis Pouya Bashivan,Kohitij Kar,James J. DiCarloscience（2019）](./%E8%A7%86%E8%A7%89%E6%99%BA%E8%83%BD/6-Neural%20population%20control%20via%20deep%20image%20synthesis.pdf)


## 语言智能
---
推荐人：吕柄江

2011年于大连理工大学获得学士学位，2017年于北京大学获得博士学位，2017年-2021年在英国剑桥大学从事博士后研究，2021年11月至今任昌平实验室研究员。
目前主要开展基于多模态脑成像技术的神经解码研究工作。

作为人类特有的高级认知功能，语言是我们思考和交流最重要的工具。通过语言，知识和信息能够穿越时空在人类社会延续、传递和扩展，构建能够像人一样理解和生成语言的机器可能会是实现通用人工智能最重要的里程碑。受益于深度学习的蓬勃发展，大规模自监督预训练语言模型（BERT，GPT3）为自然语言处理领域带来了革命性的进步，从过去令人啼笑皆非的机器翻译，到如今能够以假乱真的由机器生成的新闻报道，当前的语言模型（ DALL-E）甚至能够融合多模态信息，根据文字生成与之相符的图片。但是，这些在越来越多方面具备“媲美人类语言能力”的神经网络模型是否真的像人一样“理解”和“生成”语言？他们能否帮助我们理解人脑是如何加工语言，能否实现从神经活动中解码人的所思所想？在这一模块的分享中，我们将聚焦语言，探讨基于深度学习的语言模型和人脑对语言的表征与加工。

这篇文献是早期将神经网络应用于语言的经典工作。仅仅通过预测下一个字的训练任务，RNN模型能够获得对不同词类和语境信息的特异表征。

>[Finding structure in timeElman J.H.（1990）](./%E8%AF%AD%E8%A8%80%E6%99%BA%E8%83%BD/1-Finding%20structure%20in%20time.pdf)


这两篇文献报告了在当前的大规模自监督预训练语言模型中提取关于语言结构的句法表征。

>[A structural probe for finding syntax in word representationsHewitt J.,Manning C.D.（2019）](./%E8%AF%AD%E8%A8%80%E6%99%BA%E8%83%BD/2-A%20structural%20probe%20for%20finding%20syntax%20in%20word%20representations.pdf)


>[Emergent linguistic structure in artificial neural networks trained by self-supervisionChristopher D. Manning,Kevin Clark,John Hewitt.et al.proceedings of the national academy of sciences（2020）](./%E8%AF%AD%E8%A8%80%E6%99%BA%E8%83%BD/3-Emergent%20linguistic%20structure%20in%20artificial%20neural%20networks%20trained%20by%20self-supervision.pdf)


第两篇文献综述了深度学习语言模型对句法和语义的表征和加工。

>[Syntactic Structure from Deep LearningLinzen T.,Baroni M.（2021）](./%E8%AF%AD%E8%A8%80%E6%99%BA%E8%83%BD/4-Syntactic%20Structure%20from%20Deep%20Learning.pdf)

>[Semantic Structure in Deep LearningPavlick E.-J.（2022）](./%E8%AF%AD%E8%A8%80%E6%99%BA%E8%83%BD/5-Semantic%20Structure%20in%20Deep%20Learning)


这三篇文献从多个方面比较了从多模态脑影像数据中提取的人类语言表征与预训练语言模型的语言表征，发现能够更好地完成“预测下一个字”的语言模型，也能更好地拟合大脑进行语言加工的神经活动。

>[The neural architecture of language: Integrative modeling converges on predictive processingMartin Schrimpf,Idan Asher Blank,Greta Tuckute.et al.proceedings of the national academy of sciences（2021）](./%E8%AF%AD%E8%A8%80%E6%99%BA%E8%83%BD/6-The%20neural%20architecture%20of%20language-%20Integrative%20modeling%20converges%20on%20predictive%20processing.pdf)

>[Shared computational principles for language processing in humans and deep language modelsGoldstein, Ariel,Zada.et al.nature neuroscience（2022）](./%E8%AF%AD%E8%A8%80%E6%99%BA%E8%83%BD/7-Shared%20computational%20principles%20for%20language%20processing%20in%20humans%20and%20deep%20language%20models.pdf)

>[Brains and algorithms partially converge in natural language processingCaucheteux, Charlotte,King.et al.communications biology（2022）](./%E8%AF%AD%E8%A8%80%E6%99%BA%E8%83%BD/../语言智能/8-Brains%20and%20algorithms%20partially%20converge%20in%20natural%20language%20processing.pdf)


这两篇文献通过神经网络模型直接从大脑的神经活动中解码受试者想说的话。

>[Neuroprosthesis for decoding speech in a paralyzed person with anarthriaMoses D.A.,Metzger S.L.,Liu J.R..et al.（2021）](./%E8%AF%AD%E8%A8%80%E6%99%BA%E8%83%BD/9-Neuroprosthesis%20for%20decoding%20speech%20in%20a%20paralyzed%20person%20with%20anarthria.pdf)

>[Speech synthesis from neural decoding of spoken sentencesAnumanchipalli, Gopala K.,Chartier.et al.nature（2019）](./%E8%AF%AD%E8%A8%80%E6%99%BA%E8%83%BD/../语言智能/10-Speech%20synthesis%20from%20neural%20decoding%20of%20spoken%20sentences.pdf)
## 学习智能
---
推荐人：柳昀哲

2016 年于北京师范大学获得硕士学位，2020年于伦敦大学学院获得博士学位，2020年-2021年在牛津大学从事博士后 研究。2021年至今任认知神经科学与学习国家重点实验室 & 北京脑科学与类脑研究中心研究员。
柳昀哲实验室致力于解析人类高级智能行为的计算和神经机制（目前主要关注认知地图的形成和发展）和开发新型的神经编解码模型和脑机接口，为脑疾病和精神疾病的诊疗与调控提供新的手段。

从心理学的角度看，人类智能在最大化未来的奖赏收益，最小化未来的损失这一原则下，通过与外界环境交互产生。强化学习模型（RL，Reinforcement Learning）提供了一个很好的框架去描述与解释学习、决策等人类行为。在这个模块的分享中，主要从强化学习的角度去建模人类行为，并试图探究在神经元层面对应的计算过程。进一步地，也会探讨对人类前额叶进行建模的元学习模型。

这篇文献是第一篇deepRL，展现了结合深度学习网络的表征能力和强化学习算法的灵活学习能力后的潜力，其中experience replay的运用在分离当前经验对学习的作用上非常关键。

>[Human-level control through deep reinforcement learningVolodymyr Mnih1 *, Koray Kavukcuoglu1 *, David Silver1 *.et al.](./%E5%AD%A6%E4%B9%A0%E6%99%BA%E8%83%BD/1-Human-level%20control%20through%20deep%20reinforcement%20learning.pdf)


这篇文献基于强化学习理论，给出了replay的计算模型，normative model，这个model可以解释很多老鼠身上replay的现象。

>[Prioritized memory access explains planning and hippocampal replayMattar, Marcelo G.,Daw.et al.nature neuroscience（2018）](./%E5%AD%A6%E4%B9%A0%E6%99%BA%E8%83%BD/2-Prioritized%20memory%20access%20explains%20planning%20and%20hippocampal%20replay.pdf)


这篇文献同时从算法和神经层面提出了表征式强化学习的机制，简单而言，即不同 neuron 编码不同的value，所有neuron一起编码了value的分布 - 这种表征可以极大地提高人工智能体的performance，也是生物体多巴胺神经元的表征机制。

>[Distributional reinforcement learning in the brainLowet Adam S.（2020）](./%E5%AD%A6%E4%B9%A0%E6%99%BA%E8%83%BD/3-Distributional%20reinforcement%20learning%20in%20the%20brain.pdf)


这篇文献通过区别 slow learning 和 fast learning, 提出meta learning system 可以作为前额叶的计算机理的研究框架 - 对解决 learning to learn 等泛化问题，提供了新的解决思路。

>[Prefrontal cortex as a meta-reinforcement learning systemWang J.X.,Kurth-Nelson Z.,Kumaran D..et al.（2018）](./%E5%AD%A6%E4%B9%A0%E6%99%BA%E8%83%BD/4-Prefrontal%20cortex%20as%20a%20meta-reinforcement%20learning%20system.pdf)


这篇文献是发表在Cell上的一篇关于机器学习模型揭示大脑怎样整合空间记忆与关系记忆的文章。
（解读文章： https://mp.weixin.qq.com/s/FuB_eSzlD2lwO3aZiEnagQ）

>[The Tolman-Eichenbaum Machine: Unifying Space and Relational Memory through Generalization in the Hippocampal FormationJames C.R. Whittingtoncell（2020）](./%E5%AD%A6%E4%B9%A0%E6%99%BA%E8%83%BD/5-%20The%20Tolman-Eichenbaum%20machine-%20unifying%20space%20and%20relational%20memory%20through%20generalization%20in%20the%20hippocampal%20formation.pdf)



这篇文献基于海马-内嗅皮层的架构和学习机理，提出了学习和泛化的计算模型，将目前预训练语言模型Transformer架构的表征方式与人类海马体的表征方式在数学上进行关联比较，为解释Transformer的biological plausibility提供崭新视角。

>[Relating transformers to models and neural representations of the hippocampal formationJames C. R. Whittington,Joseph Warren,Timothy E. J. BehrensarXiv（2022）](./%E5%AD%A6%E4%B9%A0%E6%99%BA%E8%83%BD/6-Relating%20transformers%20to%20models%20and%20neural%20representations%20of%20the%20hippocampal%20formation.pdf)


这篇文献提出一种类似于HMM的海马计算模型，相对于TEM，其好处是更快的学习；坏处是难以分析结构泛化的机制和神经表征。
>[Clone-structured graph representations enable flexible learning and vicarious evaluation of cognitive mapsDileep George,Rajeev V. Rikhye,Nishad Gothoskar.et al.nature communications（2021）](./%E5%AD%A6%E4%B9%A0%E6%99%BA%E8%83%BD/7-Clone-structured%20graph%20representations%20enable%20flexible%20learning%20and%20vicarious%20evaluation%20of%20cognitive%20maps.pdf)