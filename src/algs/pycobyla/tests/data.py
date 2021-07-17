import numpy as np

BAD_WAVES_FIELD_START_X = np.array(
    (
        (0.98339995155923931591246, 0.39446041597099967823681),
        (0.98339995155923931591246, 0.39446041597099967823681),
        (0.95591205338974805094665, -0.36856458232465127977662),
        (0.97530020755164614776334, 0.76426749874256905137315),
        (0.83423062777935252931627, 0.95234139277480500673789),
        (-0.87439520330653786039932, 0.96539099492240243449714),
        (-0.79177056891242192371294, 0.95053853257629028483677), 
        (0.96794841398793063369510, 0.78593216088240724914726),
        (0.96696354360205538647222, -0.41931914034568973370654),
        (-0.31321550603449077598839, 0.97609160329994826277300),
        (0.97790962554607929746453, -0.26520497972541767772725),
        (0.97440277413151799024149, -0.76728858527760746000013),
        (0.96886712415532572073573, -0.38365156970732972041560),
        (0.95774138640839745484357, -0.36911808399218437592992),
        (0.78332253267200369073464, 0.95034937305881417302089),
        (0.99692964635842540310762, 0.57384217052600794417572),
        (0.69290576856162466867772, 0.97301845995413938084084),
        (0.10505046934571526939806, 0.95227693455608752870489),
        (0.58679939136432923696418, 0.95906590555799708930351),
        (0.33200480969480716808562, 0.97066115309959455359490),
        (0.98295483153191787195624, -0.69349840245212046596635),
        (-0.65645219001626253785275, 0.98212984777026690608182),
        (0.96140436337184875803530, 0.57559994605205377915524),
        (0.96863018717201598839495, 0.22746171511912649521037),
        (0.97787301186327479918248, -0.75972520259851639146120),
        (0.53560804292650510127771, 0.99344428412720797716418),
        (-0.78177610082653936629526, 0.95530841334910032713879),
        (-0.34773509517891199038786, 0.98330356146917341497726),
        (0.50306678613082866924344, 0.99991959289041876246529),
        (0.14994494491287424509096, 0.95832856854572012750282),
        (0.95296493992982833631800, -0.42112356720060328818533),
        (0.32292383715618333539510, 0.96459803483735928608667),
        (-0.32951277043785864862002, 0.97237533384796859259325),
        (0.97206368914864560437650, -0.25878926733123130965453),
        (0.96865009645302913021681, 0.57261802689486929196505),
        (0.68129638330991393324609, 0.95846864359039352088132),
        (0.75271784940471975211551, 0.98517837253490747606577),
        (-0.34168505076151789445760, 0.95447731960752135726977),
        (0.97109448468686565547614, 0.77191764207388668950216),
        (-0.74057121287450877744618, 0.98273853029157165472895),
        (0.95023813381517308052082, -0.27593254891002638728992),
        (0.98534952823780708186518, 0.54618710772749290427441),
        (-0.63958379230813955373947, 0.99013569241177168400725),
        (-0.34102823057773390402758, 0.98062821038061920297935),
        (0.95467742053027926374398, 0.81155806696115506682077),
        (-0.20708221589698494469189, 0.96134750151310410792860),
        (0.66988040345693367072499, 0.97302728780574221367772),
        (-0.85253577695070692499257, 0.98517095432062795623551),
        (0.14865917593671196250682, 0.96064082840459352752305),
        (0.98212449895210185424332, 0.75866827834464634072731),
        (0.98483311067484180512110, 0.58573076621186315193768),
        (0.48111983894749177359529, 0.99621072648990471876118),
        (0.99743669612067598073679, -0.47274892821166014655887),
        (-0.42243137007219799805569, 0.95510898814424605163254),
        (-0.29667122393356915033280, 0.95894141169385394007918),
        (0.11761273288242901635670, 0.97330926653381966140444),
        (-0.52509185580201411802648, 0.99810341630725374351130),
        (-0.78576915000258473753547, 0.95045673949032827287908),
        (0.95197565120907068347833, 0.62896392894599628675678),
        (-0.52457794913917288326388, 0.99826651086061679585271),
        (0.64979302514742598440023, 0.99880731891887086781878),
        (0.96721961665893529946914, 0.67238930039215771827799),
        (0.95478654121966322065873, 0.70830811688415162841181),
        (-0.88426842974774477745825, 0.95609993186508002160906),
        (-0.24292082383364133058024, 0.95768184665834055202538),
        (0.95107112495350998315757, 0.58240400077995202465786),
        (0.96870098315471886429862, 0.42231419799790059776967),
        (0.15659472959078102327624, 0.98388049785143905090479),
        (-0.77422957086245447477779, 0.98698136028422078780409),
        (0.99686384538906303021122, 0.55278544028892939721231),
        (-0.21695124670179710690832, 0.97380471046390426614892),
        (-0.34922020695790023658844, 0.97616801164099364918059),
        (0.13674964547206358567166, 0.98613492014667181173593),
        (0.62125755191189258042073, 0.98082163657867327266615),
        (0.98154222252802969528318, -0.22821942655611215933220),
        (0.98646513199887886358397, -0.34997783254204928304887),
        (-0.78842096365541380897923, 0.97994339734456770152349),
        (-0.30451393740604459914323, 0.96413421062053283883131),
        (0.69360278898911120748494, 0.97521902353628586723744),
        (0.15685551780725504045222, 0.96682863175006672484812),
        (0.30912096699023949852858, 0.95566348495241948413081),
        (0.53521880608410632440552, 0.99350800659291760297265),
        (0.26699885325343353237315, 0.95566216010603088903963),
        (-0.69816567878070712183103, 0.98342831096951122127336),
        (0.33828409964736705362043, 0.98320035604499289583202),
        (0.53968350522926833434667, 0.99319581837099457644058),
        (-0.37851626855644360247766, 0.96265406552684074092952),
        (0.61516952686133019589931, 0.97017410420155458794511),
        (-0.65579971906328737851766, 0.95942591801211252189319),
        (0.62849393383258078671361, 0.98453636169859159998907),
        (0.98119508460035942398747, -0.32017789430400811490074),
        (-0.66076089981129393358117, 0.95124902749662521017626),
        (0.14082453479662748740964, 0.98059230067169522726545),
        (-0.33343155314632522134843, 0.97438650559299522235790),
        (0.97963702503878180571917, -0.34102937846237213470602),
        (-0.87682184679889929590502, 0.96009218861789391574746),
        (0.94992807408339796637620, -0.96709088164250323060855),
        (-0.81369159834172455347812, 0.95658488879107350655318),
        (0.98331469974435559144865, -0.72642870103400758452494),
        (0.98174358150364704478363, 0.63239496136317274732619),
        (0.65348686226324170789326, 0.97413589095068386924936),
        (-0.38928715363055177434148, 0.96813925841874404198961),
        (-0.36376299303785564198677, 0.98626512368023533383621),
        (-0.79193654751107644962360, 0.95131290964158132794637),
        (0.58373664755808229998024, 0.95028677626433455394306),
        (0.34096854095256268202263, 0.97803072573811400403088),
        (-0.80022422805113468946558, 0.96907618787563287732212),
        (-0.77746180820334398475779, 0.98676909928437162022874),
        (-0.77699880441865909475041, 0.98357322288263326903746),
        (0.96376713584970419290698, 0.62686794661037148479465),
        (0.34886765688366172533108, 0.98797773106787656338668),
        (-0.45749433519861359975778, 0.99083510070935654212576),
        (-0.42418694017032176901694, 0.96148825397762993638651),
        (0.61487466242965171936419, 0.96877821996863389131249),
        (0.97898866000859086078378, 0.24191166249180295899635),
        (0.21597089136675595710813, 0.98491276320155263235279),
        (-0.88307888376186949308533, 0.98085725074654339650237),
        (0.98089246199778035162353, 0.62184117566315033620583),
        (0.98656071364524811073693, 0.35325504413269981363044),
        (0.97801955856886024776031, -0.42838715336160659852283),
        (0.10465723976276053441836, 0.96926079578918877821536),
        (0.95004929908019675188768, 0.58626589548920948757882),
        (0.97684900183808753837411, 0.33723735707654056703575),
        (0.96646085898059252983217, -0.37627888904507078571271),
        (0.97609600981544653031108, -0.21188712213685567675725),
        (0.97036782417848921333814, 0.61469550713584975021320),
        (0.97607974069021419261105, 0.77919261676799211890909),
        (0.24255056433824240258446, 0.98311418986353649174248),
        (-0.78623646413926162601626, 0.95047909258396323650686),
        (-0.34250207541258292920361, 0.95792007150048208430348),
        (0.97647199242929216822517, -0.56804066833088295851439),
        (-0.29216337783318513388053, 0.95616804749098105453697),
        (0.99683866095576201260542, 0.54756170127547854065142),
        (0.96739328113738110026532, 0.69900559163198416889884),
        (0.95164152393491319159580, -0.19107341223240736916011),
        (-0.29544511309010723643098, 0.96109297102788548983199),
        (0.98393025608942652482369, 0.56313045633720881788520),
        (0.28661727528251090824085, 0.96522966465572790362160),
        (0.42813577721733975245400, 0.99948645784164180660980),
        (-0.76811595119591435931738, 0.97285450209822799116921),
        (0.98506231409913924679245, -0.35045852870980720439320),
        (-0.68893677014118792634179, 0.98585953343897658740502),
        (0.97735736504673842439672, 0.76030377804402959895924),
        (0.64617121607285765705342, 0.98707119515433694445505),
        (-0.50746054117988470544276, 0.99776086494267612891917),
        (0.17731384633338853618056, 0.98466978460844134524166),
        (-0.79782300878220402395868, 0.95069600375955465310085),
        (0.26083791921919075029734, 0.98366265703668132047710),
        (0.96326199464607276112815, -0.25379637950740452367882),
        (0.67137136217865567289209, 0.97010542589525239343118),
        (0.98195122673503765220460, 0.46575922711774153661679),
        (0.98415397574532503810474, 0.57290911041541803250254),
        (-0.33854020301735787690234, 0.95410267215979716048935),
        (0.23762594424431249251484, 0.97964449308460022081135),
        (0.27242561114527963361809, 0.95457483178081137253912),
        (-0.30578556228291664531582, 0.97506622864964809238586),
        (0.95601604893449709798858, 0.75461474476798318100634),
        (0.97572845463816304523164, -0.22230239103923143950681),
        (0.22510524925481467661825, 0.98641647098748808097923),
        (-0.24824253456575529064310, 0.98535043665057475692493),
        (0.96186596300084281629950, 0.78035443632093559607199),
        (0.11450170680868088091131, 0.95212041217989518138154),
        (0.22122439410189587150057, 0.95769208384740922568312),
        (0.32372563622003025507468, 0.96480824168383705341512),
        (0.65844110662244559328826, 0.98001811373057678977716),
        (0.98539113818874413119886, -0.45071306326369175998536),
        (0.74918692743588222171525, 0.98723035935695180320693),
        (0.99242020495320004691564, 0.57421148798288568215753),
        (0.76364842708464153453463, 0.98671213913254640637263),
        (-0.34504510915921371427828, 0.96791342749395758993103),
        (-0.35716487457418977236046, 0.98318113423526787286733),
        (0.14242572965511235416614, 0.98174080123934670538688),
        (-0.31894678500959194877851, 0.96087429464835838466286),
        (0.99661466320267666496591, 0.38040143649255386826269),
        (-0.53403929171844466416985, 0.97952718766055402177528),
        (0.99901501166836492728862, 0.54391237006443704515846),
        (0.34190775457873345821724, 0.97920731109339831377270),
        (0.12602110699498991763789, 0.97652639991952661091545),
        (0.76629413905706678100671, 0.98507220612044577023880),
        (0.97201861867399119709887, -0.21846346766549173956662),
        (0.63273683882208731077412, 0.95588533061625446940468),
        (0.62799946274267903767452, 0.96060119464565940639034),
        (-0.65765939197108247427082, 0.95684636546951451485654),
        (-0.41184082752580808417520, 0.95738084897796227323852),
        (0.97822153331608308057810, 0.64759211940953287367506),
        (0.22936134907179583919401, 0.97096902542006424674526),
        (-0.54974651999678969538365, 0.99650049549886388078335),
        (-0.40180749616615707431322, 0.98386932072524646564204),
        (0.61935356610148262213045, 0.98064201149750696018259),
        (-0.37993085135833415399986, 0.96384820694453821054992),
        (0.67519293343364950032992, 0.98447703812826681257775),
        (0.13293709645659079754410, 0.97280320108564732883849),
        (0.99116632879984867265932, 0.54924761089056395313435),
        (0.98601763617983739784734, -0.65468819593281146751451),
        (0.98408429173545752810526, -0.34613361648288121230621),
        (0.97131472910727345571047, -0.23000001651540058489331),
        (0.10003635291629842107852, 0.95359727710442832027127),
        (0.96156970016791354360919, -0.24640481365746413899842),
        (0.27173549391116047502237, 0.95287326083169188173372),
        (0.77360831360740012208055, 0.97911021927785735208261),
        (-0.66250316538143749767187, 0.97777629189788850538889),
        (0.99907785458944986523022, -0.63266022909543950802913),
        (-0.43774780616844233271934, 0.97563099263865993293621),
        (0.97596461322041649921744, 0.77751996272385892616796),
        (0.80069377685420062285004, 0.95021681590943418704853),
        (0.97028962517055350467388, 0.43182591814230608662228),
        (0.12217052647262804931927, 0.96292990565376812561738),
        (0.98358429068248454107959, -0.37945930006684869262301),
        (-0.37996938239234778755815, 0.98433189904980999962447),
        (0.99783592627240236261343, -0.62127531797042223615790),
        (0.11251517859995430193010, 0.96950291693726575736889),
        (0.97844071692545275809039, -0.56230132396380327897134),
        (-0.72016432986297718166213, 0.96407391939805098246552),
        (0.97963072086773705926532, -0.43054148740096986891501),
        (0.95159646799771135938784, 0.78396438562845527364686),
        (-0.72321944052157727256258, 0.95155249311551104440809),
        (0.33769929504801310393702, 0.95344425215817651952932),
        (-0.89457084990368485044598, 0.95738985892976313785141),
        (0.98697764903747509102061, 0.59765381839004150243966),
        (-0.21798022701290742375591, 0.98201336945986672510855),
        (0.98458769400964518681008, 0.63786394210224250933550),
        (0.31958573458464378802546, 0.95982644343984624413224),
        (0.78340021476068200101395, 0.95283497492680013785105),
        (0.66002610219394730606268, 0.95890228919615738689686),
        (0.97394764033973380001896, -0.26333030630310072517375),
        (-0.69254290589448430637276, 0.98357209894242458858571),
        (0.19145657914806246679973, 0.95510511049833946017884),
        (0.98616078575294596753054, 0.54673770843571056765597),
        (0.98733350029368449618516, -0.38917022062961148520799),
        (-0.38168451903304623229474, 0.99520052563251826249768),
        (0.27268264841191180991586, 0.95513586489705315507592),
        (-0.34097163207547631635919, 0.97855794117189831560211),
        (0.95194870975527079437484, -0.21163957766350427469604),
        (-0.32876669679905878851400, 0.96862746314650172152483),
        (0.95107182243270393939838, -0.30168685637495573637068),
        (0.97257488223434718221938, -0.32963132320436505651173),
        (0.96359588586594013825959, -0.42308835343370132520135),
        (0.12786892300097596475439, 0.96943895084609366108452),
#        (0.94688144141574981382803, -0.98573983882119375898867),
        (0.74357810728141338074693, 0.97272203647028332440527),
        (-0.75492467792360695710840, 0.96080138683469806792914),
        (0.98006565658919519989922, 0.68106042054911686278729),
        (0.96969472186071148378517, 0.65154482069967079027606),
        (0.96825868978786444607465, 0.18455580170345720070202),
        (0.96510703557266164231976, -0.37412014249273650712269),
        (0.49382263659757064111488, 0.99870003275588947744268),
        (0.98748186218064204489053, 0.65020528671562827227604),
        (0.97218529648828777567360, 0.65607491855725230678331),
        (-0.63602799089968242718385, 0.99316000650816871342386),
        (0.95234275415265634556761, 0.62966313119673222864492),
        (0.16663053560218199500298, 0.98160960513634676338768),
        (-0.73247232216173618901678, 0.99456669430137822907056),
        (0.98700606170485905010992, -0.23627301804416078745419),
        (0.98308733339227116765358, -0.43843302307499354419917),
        (-0.20835536390546605112206, 0.97422314798230913446275),
        (0.98550326284288125577859, -0.45207984360722464067806),
        (0.97398293514005085391716, -0.66555048110854042597850),
        (0.96746234743704628833427, -0.29536498171737268769732),
        (0.95525420625660140139246, 0.65925642771776771233760),
        (-0.34047274165365748466172, 0.96500110673453698417745),
        (0.98862185987634965123050, -0.55558096478987883948264),
        (0.99221918224525262886004, -0.45378223079693014163638),
        (-0.72941435589732006583574, 0.95969876586809244045639),
        (-0.19204896912792257523961, 0.95586741974903799423657),
        (-0.72890430338851186498061, 0.95140728860371082120650),
        (0.15152336841731761651886, 0.98568254935940147198892),
        (-0.21140425305103516251393, 0.96638762131375810504608),
        (0.96264646037886358875824, 0.77636444568129303611670),
        (-0.24605385987284766891037, 0.96123841138746923817848),
        (0.97204638126635911632434, 0.66802301710901290654476),
        (0.80023584228838373633153, 0.95109868095903449258799),
        (0.14564987929266881572232, 0.98584364410172531378862),
        (-0.89887351504116752565210, 0.96736091560625125929107),
        (-0.87451900428062101511273, 0.98576809010122312670887),
        (0.96948635174897956900963, -0.21549627446750818648979),
        (0.98950312802740714168692, 0.33065981464796623257030),
        (-0.68314767161640088666275, 0.96405347420369147393160),
        (0.24103946122295005771718, 0.98101794850006807102716),
        (0.95608930473816977624324, -0.29178521959360192994382),
        (0.98913191697904578880696, -0.45463284626136135635477),
        (0.96008989911202635703091, 0.66774762999450087974651),
        (0.94687348165846163539072, 0.00143508633036959487583),
        (-0.22920931322320647893775, 0.97224609407353734802371),
        (0.16242674429714676342940, 0.97792714763388843834946),
        (-0.62792648926749872195785, 0.99398394561989311668526),
        (0.62364934314447562080375, 0.95922916481295739110635),
        (-0.71432862424579979254702, 0.96932732974999291641893),
        (0.98593046876871670569642, 0.39538518292229052342179),
        (0.23809016873743460429580, 0.97961820104364316641465),
        (0.33199565087123117379519, 0.97339973419077852057057),
        (-0.30250962155426286415150, 0.96937992866524513857485),
        (0.98567229001658662745911, 0.54656493570592323827384),
        (-0.41392847687470912809715, 0.96148803051229259075683),
        (0.95965277975885610040052, -0.41913603975738622509084),
        (-0.69040428683175836255259, 0.97808702100803990298061),
        (0.19743486761085349279199, 0.96316674793424827960564),
        (-0.72356399267734627933635, 0.95816580532755701860026),
        (-0.78311395824168195112236, 0.98524548086087171405723),
        (0.97434189148699190674563, -0.23006544801141171063819),
        (0.95387675933307303210995, -0.33992361236651835731948),
        (0.95445278860010063226582, -0.43790387799784080691268),
        (0.97921350565718379144187, -0.69215384813974090150168),
        (0.78748319810342803926062, 0.96389530798572065073415),
        (-0.56630173823744556216297, 0.98051138599035025933404),
        (0.56228936041434107728776, 0.98666563707100007896145),
        (0.99981948745838145065079, -0.47548637932386195181778),
        (0.28698034962201424136197, 0.97764408694581628189724),
        (0.98340611008005951454436, 0.76606804081138868411927),
        (0.95736006019494168661765, 0.78018477287769782968496),
        (0.97413610810335748979583, 0.23241879451162894554272),
        (0.99781085733554397698697, -0.62037951627483511884975),
        (0.58566133952407284368746, 0.95143471019862491111496),
        (-0.16602250028863019259973, 0.95152639666593663569927),
        (-0.85525488583023934197058, 0.95968428645153802669654),
        (0.97712952556935617209888, -0.76124795744107842665471),
        (-0.73607582576516983330350, 0.98744703409812850658511),
        (0.24768520108525304834757, 0.98611600764257167917037),
        (0.56417181783567249553357, 0.98124382848953062286057),
        (-0.42797712842995538906621, 0.97246086223355820976622),
        (-0.67417728397893528224927, 0.96717487994980366750042),
        (0.96754991697348113888211, 0.43080970393627615599996),
        (0.96730247324518092710832, -0.30671446990050998948618),
        (-0.69157222916897320708074, 0.98738036091970071694846),
        (0.97833652342765531351176, 0.30951091066712610455625),
        (-0.65592964583223523078459, 0.95939218481935117921466),
        (0.13307851182737606698936, 0.97565957848697193455223),
        (-0.84360594902000318739965, 0.97306180238693351647328),
        (0.97176041350460673484690, 0.42941384266382809364870),
        (-0.85964918845308035955100, 0.95330313361643637115606),
        (-0.36031026829497259100776, 0.98614986670733828511004),
        (0.95906600023737298421622, -0.55589823352266853895287),
        (0.24425403861460037724385, 0.96317744234248681145516),
        (0.98230765791824992128056, -0.63415188341557815476790),
        (0.12057004206245647282003, 0.97576107777423737310585),
        (-0.29055777815896921545402, 0.95369640620456874735567),
        (0.95476235030945777459976, -0.34007931748667963312016),
        (0.95957200608547332798537, -0.20524456920858891173509),
        (0.11437085512956146615693, 0.95389264261326678351338),
        (0.98087029561272864874866, -0.42598088318693783094204),
        (0.64484974685489038215280, 0.97960347692489535553761),
        (-0.35676635201965334331931, 0.98092396546485205455213),
        (0.99726833967879291442671, 0.37923257730822634847812),
        (-0.25928968591214363037523, 0.96944891141172861992459),
        (0.98734075362981088552772, 0.72368294759671303317816),
        (0.23818198785415112261887, 0.95802524004656963718674),
        (-0.31889908574632142013172, 0.96255903680777765707433),
        (-0.88549611717330556359684, 0.96697144189841854888812),
        (-0.78427425798421634972613, 0.98234930561399846915549),
        (-0.78447927434680253533372, 0.95212309318904431165720),
        (0.95922544065764681064934, -0.21912960502285971919889),
        (0.57956178857717222108192, 0.96922728920960099507909),
        (0.99271990213099714495115, -0.58789633840609423209855),
        (0.12886998620943113103010, 0.96874387703215325551298),
        (0.97699970556342363714464, -0.31469876434367050066498),
        (-0.24399622427846323624578, 0.98155004733983508558026),
        (0.96694072712906020683477, 0.74962601881744173049071),
        (0.97156293992810249804393, 0.67166139610082220556819),
        (0.64716452922365763633650, 0.97849260288180661682134),
        (0.15306682955385908506685, 0.96923367996330833662455),
        (-0.82826674127524002599898, 0.98234142389066247957885),
        (0.16065942464623539365220, 0.97691007587801670375427),
        (0.96558448471517954736498, 0.43209440912295749015470),
        (0.95084014387857096473056, -0.32791037240554632425926),
        (-0.22548962803666317000761, 0.97829092315656041023431),
        (0.12069027714548719032450, 0.97596564727924151050331),
        (0.11717305807976785558822, 0.95134952983594844866388),
        (-0.71450574299565872138373, 0.96921204899028690071816),
        (0.15852645704971313200815, 0.97237644725017613467344),
        (0.96733496798805851391023, -0.38773559279655600562364),
        (-0.87443829047970567103221, 0.96325693226381647882306),
        (0.96612302884881584574828, 0.69926248398617141255329),
        (0.56893268642237782017901, 0.97200942551917957068497),
        (0.97819237628186161970234, 0.57148826193575175125261),
        (0.23961437903630944390443, 0.95364310988041833816453),
        (0.98246984333894959995348, 0.77191729223930050096669),
#        (0.94804926764635699676376, 0.02329135435529239295249),
        (0.95142177375824088514378, 0.67341301791312280222712),
        (-0.75575997792145099829497, 0.97015953967860957263269),
        (0.57395240319986928767548, 0.98642083572560101956128),
        (-0.78588998377354757352009, 0.95397597082942131585526),
        (-0.33000653801123380759464, 0.95814509382365375955715),
        (-0.44078847994404979360183, 0.97786895911428550753897),
        (0.98343273697289457224713, 0.65798067788040404124672),
        (0.77507991001792597707265, 0.96628911141822193542339),
        (0.97646234899064698886662, -0.41076475730981010237031),
        (-0.44100563739049913891677, 0.97783276275369468422127),
        (0.77177746012091730243299, 0.96811651498399298176878),
        (0.16948898070915641156375, 0.98310285596460689205855),
        (-0.80858312991319070128782, 0.95384163330945015424334),
        (0.20505091486739357442559, 0.97178789430397749526946),
        (-0.38450477717455888715392, 0.97039215352602092856671),
        (-0.88935933332692274078113, 0.96398604939269438851568),
        (-0.31303249910005193079598, 0.95198373927046975317978),
        (-0.63633001377929754305285, 0.99324535284580717231506),
        (-0.77174747220743200593063, 0.97272481767888296921853),
        (0.21549579295739751394478, 0.95317053544225771588572),
        (-0.84085565076729928968291, 0.97210247775992386642940),
        (-0.42465512755770440378456, 0.95838239516543444196373),
        (0.54640386932697593280750, 0.98816997763113789687850),
        (0.61493121389272187293784, 0.97029871627186103921758),
        (-0.85943249574570401705387, 0.97916246893434633058462),
        (0.97647823953347456793495, -0.76108594641756321941273),
        (0.23803574528993598491411, 0.97872831562956918993734),
        (0.97432735536107584195520, 0.64994904474512127379171),
        (0.78926862029980093815595, 0.97945081135596701038537),
        (0.95276652338878808912170, 0.67090137988717457595556),
        (0.95451091109160857151039, 0.71236857602573655512401),
        (0.79901684734372668827973, 0.95479540744509083793901),
        (-0.57208307128150392983912, 0.99850509954172705917586),
        (0.96421426165533463681356, 0.57365046881749282015051),
        (0.97394312750719636007091, -0.34951831446354253429831),
        (-0.85506941056390606270554, 0.98134967903768366426220),
        (0.99729782276159073539645, -0.61994522612881497458659),
        (0.78999523993396403298561, 0.96482808299866062995420),
        (-0.86822295171146168790699, 0.97149013860586652668871),
        (-0.31430156586104485150202, 0.95421972850441916236264),
        (0.67836348991078465608950, 0.98462599723498311909964),
        (0.99965075624532895304242, -0.60527813001824748440072),
        (0.99745921115063707063086, 0.35875942506375069207536),
        (0.98335937390195593543751, -0.37895602382179749412217),
        (-0.41880299193967762683144, 0.95301804486766394930441),
        (-0.75734613865582733893689, 0.96345695282999299635662),
        (0.98514263847234095905492, -0.24609342837760972422245),
        (0.95712843816375081651415, 0.56312598406787217619751),
        (0.97988446086818226099524, 0.39252988817824019385228),
        (0.26588481407282626989286, 0.98092474139388330378608),
        (-0.33439881461775944337944, 0.96606819537641541728590),
        (0.98259905819254966807819, -0.31550432099937086860564),
        (0.46669222401957699553066, 0.97776182265744204791247),
        (0.98026730483663460091748, -0.58784970201318564875237),
        (0.35866818323618443464795, 0.99113625759802892467576),
        (-0.64929037363152564132918, 0.99780874806086727879517),
        (-0.33051980107821865573214, 0.97018615627567350578886),
        (0.78503943301226297712958, 0.96872295376472061789741),
        (0.77120023980102314631324, 0.96719522963988113772871),
        (-0.74988663585206438710884, 0.99979047611400573281060),
        (-0.75494149116723718861977, 0.98364015143051308776023),
        (-0.66979370403429583369359, 0.97180608749171537574796),
        (0.99252247873655763399370, 0.40898436683948857783832),
        (-0.72155199916381618230332, 0.98134620326831423220426),
        (0.78355700277888606919419, 0.95551082149720878433641),
        (0.18760095965385437111195, 0.95351785371303643401575),
        (0.33509298822898392344882, 0.97035012623841510048806),
        (0.62542050320242004168847, 0.96561008157839589571836),
        (0.75930042056025737196023, 0.95431925633290548560694),
        (0.69686748134751907990392, 0.96955052900524330006249),
        (-0.75988289494414451574755, 0.96404498431614027431635),
        (-0.86135043500843622155116, 0.95044071432231080898134),
        (0.68500646826978273118414, 0.97738377348695171242809),
        (-0.65631753120298030879098, 0.95774106545372639232028),
        (0.99181641234874473589400, 0.36046611695167185551725),
        (0.96773219611322902622419, -0.25454811648895314668550),
        (0.33353648140313119085931, 0.97761893450111703174343),
        (-0.83461163632427726177809, 0.95185708966941851016941),
        (0.98473998744335333732636, 0.62817420234702936454596),
        (0.96684244394735374683592, -0.22564199735490264586701),
        (0.80388318332706432123302, 0.98428366616778140141264),
        (0.97413112836535553640260, 0.66563073558230190229779),
        (0.95461940403813039246472, -0.41594555435567004408881),
        (-0.37801428735584208240539, 0.95556359769564291894994),
        (0.95553023132343706258496, -0.21709912557096688878744),
#        (0.94750328598303457106056, -0.98721532010223267405991),
        (0.95455187415000342099347, -0.24194923504170318118156),
        (0.96871258730173503970207, 0.77548319648804331372105),
        (0.68046921316232089615994, 0.97873306992885256150316),
        (0.79224433641149372142820, 0.96002363196310125381672),
        (0.14593316437384062922433, 0.95636376996089755841979),
        (-0.51160529566553591962474, 0.99841082013973236897186),
        (0.98406684471850702777829, -0.24609080705049213300128),
        (-0.51201360404198892339878, 0.99706041233676345747483),
        (0.98601841819393487575951, -0.36154426511099946317529),
        (0.98132430959714289642193, 0.41644003660875972272493),
        (-0.88873325180698525826983, 0.97689635716005351895319),
        (-0.76941083222215778114617, 0.97053335735755319468865),
        (0.98141989721147027125880, -0.35429643053444959122089),
        (0.12877416479060133092105, 0.97003996071034226389429),
        (0.98892336042831519016261, 0.35433624328111013035425),
        (0.78302020248350578945917, 0.97249826912623293928561),
        (0.96344070213840060645794, -0.24743404183931327899870),
        (0.99635415177993613689011, 0.37062220839446147202523),
        (0.27138294922170946854578, 0.95051468381118020545273),
        (0.99234263217743157170503, 0.54324254599929155951088),
        (0.95860820108255118121576, 0.63103065797621171739706),
        (-0.68083184175022148920675, 0.95990534067926325256792),
        (-0.19856609976478489798524, 0.96243057599140158231421),
        (0.97754852901227828887443, 0.33673992911235983704898),
        (0.24449562017207004949171, 0.98223531606341407673710),
        (0.95337754130008245390115, -0.41522467156799436338588),
        (0.65799492689271765755166, 0.96496943877375485065784),
        (0.95889533242126856471543, -0.42283385872451906273284),
        (0.97624915401540302006822, 0.24120822787883988702617),
        (-0.22020896287106306310477, 0.95821403591683318445860),
        (0.69628608255374979663088, 0.97022237843299774517902),
        (-0.23735349865487020970534, 0.97814029071842090168332),
        (0.68393184408266050056113, 0.95394451580976724613947),
        (0.94803276663477697994153, 0.01120191869367204162700),
        (-0.76174963261279504678214, 0.98085044120214237572952),
        (0.96944397526111503182733, -0.38343253768065244102559),
        (0.95559393675655845967754, 0.75427041058798960371234),
        (-0.22584606745300939145693, 0.96750315259606844975337),
        (0.98639969671482896629300, -0.34972715114972929839610),
        (-0.36947889321066118561987, 0.95000176135741787319944),
        (-0.16813299307733164944523, 0.95029767156610822631535),
        (0.78304224019133106260426, 0.97913575944373776316354),
        (-0.77875355815379321278158, 0.98444972255496066004810),
        (0.31512622322060934720866, 0.96174468122618406162871),
        (-0.42078341509150862798094, 0.95583965594015851685583),
        (-0.23200820337268468485092, 0.97218545637379327750693),
    )
)
