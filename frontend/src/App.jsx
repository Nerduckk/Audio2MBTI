import { useState, useEffect, useMemo } from 'react'
import axios from 'axios'
import Papa from 'papaparse' // Import thư viện đọc CSV

// TỪ ĐIỂN GỢI Ý NHẠC THEO 16 NHÓM MBTI
const mbtiMusicDB = {
  "INTJ": [{ title: "Do I Wanna Know?", artist: "Arctic Monkeys", vibe: "Bí ẩn, Sâu sắc" }, { title: "Starboy", artist: "The Weeknd", vibe: "Tự tin, Độc lập" }, { title: "Hoàng Hôn Tháng 8", artist: "Bức Tường", vibe: "Suy ngẫm" }],
  "INTP": [{ title: "Space Oddity", artist: "David Bowie", vibe: "Vũ trụ, Lạ lùng" }, { title: "Creep", artist: "Radiohead", vibe: "Nội tâm, Phức tạp" }, { title: "Thắc Mắc", artist: "Thịnh Suy", vibe: "Tò mò, Chill" }],
  "ENTJ": [{ title: "Believer", artist: "Imagine Dragons", vibe: "Quyết tâm, Uy lực" }, { title: "Stronger", artist: "Kanye West", vibe: "Tham vọng" }, { title: "Vinh Quang Đang Chờ Ta", artist: "SpaceSpeakers", vibe: "Lãnh đạo" }],
  "ENTP": [{ title: "Don't Stop Me Now", artist: "Queen", vibe: "Bùng nổ, Hỗn loạn" }, { title: "Can't Hold Us", artist: "Macklemore", vibe: "Năng lượng" }, { title: "Thủ Đô Cypher", artist: "RPT", vibe: "Phá cách, Sáng tạo" }],
  "INFJ": [{ title: "Somewhere Only We Know", artist: "Keane", vibe: "Sâu lắng, Hoài niệm" }, { title: "Sign of the Times", artist: "Harry Styles", vibe: "Thấu cảm" }, { title: "Một Cõi Đi Về", artist: "Trịnh Công Sơn", vibe: "Triết lý" }],
  "INFP": [{ title: "Yellow", artist: "Coldplay", vibe: "Lãng mạn, Ấm áp" }, { title: "Sweater Weather", artist: "The Neighbourhood", vibe: "Mơ mộng" }, { title: "Nàng Thơ", artist: "Hoàng Dũng", vibe: "Ngọt ngào, Nhẹ nhàng" }],
  "ENFJ": [{ title: "Fix You", artist: "Coldplay", vibe: "Chữa lành, Bao dung" }, { title: "Just The Way You Are", artist: "Bruno Mars", vibe: "Cổ vũ" }, { title: "Hai Mươi Hai", artist: "Amee", vibe: "Truyền cảm hứng" }],
  "ENFP": [{ title: "Viva La Vida", artist: "Coldplay", vibe: "Tự do, Rực rỡ" }, { title: "Watermelon Sugar", artist: "Harry Styles", vibe: "Vui tươi" }, { title: "Đi Về Nhà", artist: "Đen Vâu", vibe: "Kết nối, Cảm xúc" }],
  "ISTJ": [{ title: "Take It Easy", artist: "Eagles", vibe: "Bình ổn, Đáng tin" }, { title: "Hotel California", artist: "Eagles", vibe: "Cổ điển" }, { title: "Chuyện Của Mùa Đông", artist: "Hà Anh Tuấn", vibe: "Chân thành, Sâu sắc" }],
  "ISFJ": [{ title: "Perfect", artist: "Ed Sheeran", vibe: "Chung thủy, Ấm áp" }, { title: "A Thousand Years", artist: "Christina Perri", vibe: "Tận tụy" }, { title: "Hơn Cả Yêu", artist: "Đức Phúc", vibe: "Dịu dàng, Chăm sóc" }],
  "ESTJ": [{ title: "Eye of the Tiger", artist: "Survivor", vibe: "Kỷ luật, Mục tiêu" }, { title: "Thunder", artist: "Imagine Dragons", vibe: "Kiên định" }, { title: "Tiến Lên Việt Nam Ơi", artist: "Sơn Tùng M-TP", vibe: "Hừng hực, Quy củ" }],
  "ESFJ": [{ title: "Shape of You", artist: "Ed Sheeran", vibe: "Gắn kết, Nhộn nhịp" }, { title: "Sugar", artist: "Maroon 5", vibe: "Ngọt ngào, Đám đông" }, { title: "Gieo Quẻ", artist: "Hoàng Thùy Linh", vibe: "Vui vẻ, Lễ hội" }],
  "ISTP": [{ title: "Lose Yourself", artist: "Eminem", vibe: "Thực tế, Sắc bén" }, { title: "Smells Like Teen Spirit", artist: "Nirvana", vibe: "Nổi loạn ngầm" }, { title: "Chìm Sâu", artist: "MCK", vibe: "Cool ngầu, Chill" }],
  "ISFP": [{ title: "Golden Hour", artist: "JVKE", vibe: "Nghệ thuật, Nhạy cảm" }, { title: "ocean eyes", artist: "Billie Eilish", vibe: "Mê đắm, Thẩm mỹ" }, { title: "Bước Qua Nhau", artist: "Vũ.", vibe: "Chậm rãi, Thơ mộng" }],
  "ESTP": [{ title: "Uptown Funk", artist: "Bruno Mars", vibe: "Tâm điểm, Hiện tại" }, { title: "Blinding Lights", artist: "The Weeknd", vibe: "Tốc độ, Hành động" }, { title: "Nước Hoa", artist: "B Ray", vibe: "Sành điệu, Chơi bời" }],
  "ESFP": [{ title: "Levitating", artist: "Dua Lipa", vibe: "Lấp lánh, Tận hưởng" }, { title: "24K Magic", artist: "Bruno Mars", vibe: "Tiệc tùng" }, { title: "Bo Xì Bo", artist: "Hoàng Thùy Linh", vibe: "Bắt tai, Nhảy múa" }]
};

const FloatingParticles = () => {
  const [particles, setParticles] = useState([]);
  useEffect(() => {
    const numParticles = 30;
    const newParticles = Array.from({ length: numParticles }).map((_, i) => {
      const columnWidth = 100 / numParticles;
      const baseLeft = i * columnWidth;
      const randomOffset = Math.random() * columnWidth;
      return {
        id: i,
        left: `${baseLeft + randomOffset}%`,
        size: `${Math.random() * 20 + 10}px`,
        animationDuration: `${Math.random() * 15 + 15}s`,
        animationDelay: `-${Math.random() * 15}s`,
        opacity: Math.random() * 0.3 + 0.1,
      };
    });
    setParticles(newParticles);
  }, []);

  return (
    <div className="fixed inset-0 z-[-1] overflow-hidden pointer-events-none">
      {particles.map((p) => (
        <div
          key={p.id}
          className="absolute bottom-[-50px] bg-gradient-to-tr from-mistral-yellow to-mistral-orange"
          style={{
            left: p.left, width: p.size, height: p.size, opacity: p.opacity,
            animation: `floatUp ${p.animationDuration} linear ${p.animationDelay} infinite`
          }}
        />
      ))}
    </div>
  );
};

function App() {
  const [url, setUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  
  // STATE CHỨA DỮ LIỆU CSV
  const [mbtiDatabase, setMbtiDatabase] = useState([]);

  // Load file mbti.csv ngay khi vừa vào web
  useEffect(() => {
    Papa.parse('/mbti.csv', {
      download: true,
      header: true, // Lấy dòng đầu tiên làm key (mbti, role, movie, img_url)
      skipEmptyLines: true,
      complete: (results) => {
        setMbtiDatabase(results.data);
      }
    });
  }, []);

  const handlePredict = async () => {
    if (!url) {
      setError('Vui lòng nhập link YouTube / Spotify!')
      return
    }
    setLoading(true)
    setError('')
    setResult(null)

    try {
      const response = await axios.post('http://localhost:3000/api/predict', { url })
      setResult(response.data.data)
    } catch (err) {
      setError('Có lỗi xảy ra. Hãy đảm bảo Backend (python main.py) đang chạy!')
    } finally {
      setLoading(false)
    }
  }

  const top1MBTI = result ? result.top3[0].mbti : null;
  const recommendedSongs = top1MBTI ? mbtiMusicDB[top1MBTI] || mbtiMusicDB["INFP"] : [];

  // Logic: Bốc ngẫu nhiên 1 nhân vật trong CSV có cùng MBTI với Top 1
  const characterMatch = useMemo(() => {
    if (!top1MBTI || mbtiDatabase.length === 0) return null;
    
    // Lọc ra các nhân vật cùng MBTI và có link ảnh hợp lệ
    const matches = mbtiDatabase.filter(
      row => row.mbti === top1MBTI && row.role && row.img_url && row.img_url.length > 10
    );
    
    if (matches.length === 0) return null;
    // Lấy random 1 nhân vật trong danh sách
    return matches[Math.floor(Math.random() * matches.length)];
  }, [top1MBTI, mbtiDatabase, result]); // Render lại nhân vật mỗi khi có result mới

  return (
    <div className="min-h-screen p-6 md:p-12 flex flex-col items-center relative">
      <FloatingParticles />

      {/* HEADER */}
      <div className="w-full max-w-7xl mb-12 text-center md:text-left">
        <h1 className="text-6xl md:text-8xl font-bold tracking-tighter text-mistral-orange leading-none mb-4">
          Audio2MBTI.
        </h1>
        <p className="text-xl md:text-2xl text-mistral-black tracking-tight">
          Khám phá tính cách ẩn sau Playlist của bạn bằng AI.
        </p>
      </div>

      {/* INPUT AREA */}
      <div className="w-full max-w-7xl flex flex-col md:flex-row gap-4 mb-12">
        <input 
          type="text" 
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="Dán link YouTube / Spotify vào đây..." 
          className="flex-1 px-6 py-5 text-lg bg-white border-2 border-mistral-black rounded-none outline-none focus:border-mistral-orange placeholder-gray-400 transition-colors shadow-sm"
        />
        <button 
          onClick={handlePredict}
          disabled={loading}
          className="px-10 py-5 text-lg font-bold text-white bg-mistral-black rounded-none hover:bg-mistral-orange transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'ĐANG KẾT NỐI...' : 'PHÂN TÍCH NGAY'}
        </button>
      </div>

      {error && (
        <div className="w-full max-w-7xl p-4 mb-8 bg-red-100 border-l-4 border-red-500 text-red-700 rounded-none font-bold">
          {error}
        </div>
      )}

      {/* LOADING TERMINAL */}
      {loading && (
        <div className="w-full max-w-7xl flex justify-center">
           <TerminalLoader />
        </div>
      )}

      {/* RESULT AREA: ĐÃ NÂNG CẤP LÊN GRID 3 CỘT (max-w-7xl) */}
      {!loading && result && (
        <div className="w-full max-w-7xl grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 animate-[fadeIn_0.5s_ease-out]">
          
          {/* CỘT 1: KẾT QUẢ TOP 3 */}
          <div className="flex flex-col gap-6">
            <h2 className="text-3xl font-bold tracking-tight border-b-4 border-mistral-black pb-2 mb-2">
              KẾT QUẢ TOP 3
            </h2>
            {result.top3.map((item, index) => (
              <div key={index} className="bg-mistral-surface shadow-golden p-6 border border-[#e5d8ae] rounded-none flex flex-col gap-2 relative overflow-hidden transition-transform hover:-translate-y-1 hover:shadow-xl">
                <div className="absolute left-0 top-0 bottom-0 w-2 bg-gradient-to-b from-mistral-yellow via-mistral-amber to-mistral-orange"></div>
                <div className="flex justify-between items-end">
                  <h3 className="text-5xl font-bold tracking-tighter text-mistral-black">{item.mbti}</h3>
                  <span className="text-2xl font-bold text-mistral-orange">{item.percent}%</span>
                </div>
                <p className="text-lg font-bold text-gray-700 mt-2 border-t border-[#e5d8ae] pt-2">{item.label}</p>
              </div>
            ))}
          </div>

          {/* CỘT 2: CHỈ SỐ TÂM LÝ */}
          <div className="flex flex-col">
            <h2 className="text-3xl font-bold tracking-tight border-b-4 border-mistral-black pb-2 mb-6 uppercase">
              Chỉ số tâm lý
            </h2>
            {/* Thêm hover:bg-mistral-surface và transition-colors vào container chính */}
            <div className="bg-white hover:bg-mistral-surface transition-colors duration-300 border-2 border-mistral-black p-8 rounded-none shadow-[8px_8px_0px_0px_rgba(31,31,31,1)] flex flex-col gap-10 flex-1 relative z-10 group/traits">
              <TraitBar leftLabel="Extrovert (E)" rightLabel="Introvert (I)" leftValue={result.traits.E} rightValue={result.traits.I} />
              <TraitBar leftLabel="Sensing (S)" rightLabel="Intuitive (N)" leftValue={result.traits.S} rightValue={result.traits.N} />
              <TraitBar leftLabel="Thinking (T)" rightLabel="Feeling (F)" leftValue={result.traits.T} rightValue={result.traits.F} />
              <TraitBar leftLabel="Judging (J)" rightLabel="Perceiving (P)" leftValue={result.traits.J} rightValue={result.traits.P} />
            </div>
          </div>

          {/* CỘT 3 (MỚI): NHÂN VẬT ĐỒNG ĐIỆU */}
          <div className="flex flex-col">
             <h2 className="text-3xl font-bold tracking-tight border-b-4 border-mistral-black pb-2 mb-6">
              BẢN NGÃ ĐIỆN ẢNH
             </h2>
             {characterMatch ? (
               // Card Nhân Vật Phong Cách Mistral Cực Chiến
               // Card Nhân Vật Phong Cách Sáng (Mistral Light)
               <div className="bg-white border-2 border-mistral-black rounded-none shadow-[8px_8px_0px_0px_rgba(250,82,15,1)] flex flex-col relative overflow-hidden flex-1 group">
                  {/* Ảnh nhân vật - Nền sáng */}
                  <div className="h-64 md:h-72 w-full relative overflow-hidden bg-mistral-surface">
                     <img 
                       src={characterMatch.img_url} 
                       alt={characterMatch.role} 
                       className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-700"
                       onError={(e) => { e.target.src = 'https://via.placeholder.com/400x500/fff0c2/fa520f?text=NO+IMAGE' }}
                     />
                     <div className="absolute inset-0 bg-gradient-to-t from-white via-white/40 to-transparent"></div>
                  </div>
                  
                  {/* Thông tin */}
                  <div className="p-6 flex flex-col justify-end z-10 -mt-12 relative pointer-events-none">
                     <span className="bg-mistral-black text-white font-bold px-3 py-1 self-start mb-3 text-sm tracking-widest uppercase border-2 border-mistral-orange shadow-[2px_2px_0px_0px_rgba(250,82,15,1)]">
                        Hệ tư tưởng {top1MBTI}
                     </span>
                     <h3 className="text-4xl font-bold text-mistral-black tracking-tighter leading-tight mb-2">
                        {characterMatch.role}
                     </h3>
                     <p className="text-mistral-orange font-bold text-lg tracking-tight uppercase">
                        ▶ Phim: {characterMatch.movie}
                     </p>
                  </div>
               </div>
             ) : (
               // Nếu chưa load xong file CSV hoặc không tìm thấy
               <div className="bg-[#111] border-2 border-mistral-black rounded-none shadow-[8px_8px_0px_0px_rgba(31,31,31,1)] flex flex-col items-center justify-center flex-1 p-8 text-center">
                 <div className="w-12 h-12 border-4 border-mistral-orange border-t-transparent rounded-full animate-spin mb-4"></div>
                 <p className="text-mistral-amber font-mono font-bold">Đang quét kho lưu trữ nhân vật...</p>
               </div>
             )}
          </div>

          {/* <div className="col-span-1 md:col-span-2 lg:col-span-3 border-t-4 border-mistral-black mt-8 pt-12"></div> */}
          <br/>
          <br/>

          {/* KHỐI GỢI Ý NHẠC (Full 3 cột) */}
          <div className="col-span-1 md:col-span-2 lg:col-span-3 flex flex-col gap-6">
             <div className="flex items-end justify-between border-b-4 border-mistral-black pb-2 mb-4">
                <h2 className="text-3xl font-bold tracking-tight uppercase">
                  Playlist dành cho <span className="text-mistral-orange">{top1MBTI}</span>
                </h2>
             </div>
             <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {recommendedSongs.map((song, idx) => {
                  // Tự động tạo link tìm kiếm trên YouTube
                  const youtubeSearchUrl = `https://www.youtube.com/results?search_query=${encodeURIComponent(song.title + ' ' + song.artist)}`;
                  
                  return (
                    <div key={idx} className="bg-white hover:bg-mistral-surface p-6 border-2 border-mistral-black flex flex-col justify-between group transition-colors rounded-none shadow-[4px_4px_0px_0px_rgba(250,82,15,1)]">
                       <div>
                         <div className="text-mistral-orange text-sm font-mono font-bold mb-3 tracking-widest uppercase">
                            ▶ VIBE: {song.vibe}
                         </div>
                         <h3 className="text-2xl font-bold text-mistral-black tracking-tight leading-tight mb-2 group-hover:text-mistral-orange transition-colors">
                           {song.title}
                         </h3>
                         <p className="text-gray-600 font-bold mb-6">
                           {song.artist}
                         </p>
                       </div>
                       
                       {/* Nút bấm nhảy sang tab YouTube */}
                       <div className="flex justify-start">
                         <a 
                           href={youtubeSearchUrl}
                           target="_blank" 
                           rel="noopener noreferrer"
                           className="inline-flex items-center gap-2 bg-mistral-black text-white px-5 py-2.5 font-bold hover:bg-mistral-orange transition-colors border-2 border-transparent hover:border-mistral-black"
                         >
                           NGHE THỬ <span className="text-xl">↗</span>
                         </a>
                       </div>
                    </div>
                  );
                })}
             </div>
          </div>

        </div>
      )}
    </div>
  )
}

function TerminalLoader() {
  const [completedLines, setCompletedLines] = useState([]);
  const [currentText, setCurrentText] = useState("");

  const allLines = [
    "> Khởi tạo động cơ AI Mistral...",
    "> Đang cào metadata từ Playlist... [OK]",
    "> Đang trích xuất đặc trưng Log-Mel Spectrogram...",
    "> Tải 4 mô hình XGBoost lên RAM... [OK]",
    "> Đọc hiểu lời bài hát và định vị Vibe...",
    "> Tính toán ma trận xác suất MBTI...",
    "> Quét cơ sở dữ liệu nhân vật điện ảnh...",
    "> Tổng hợp kết quả cuối cùng..."
  ];

  useEffect(() => {
    let lineIdx = 0, charIdx = 0, typingTimer;
    
    const typeWriter = () => {
      // Dừng lại nếu đã gõ hết mảng
      if (lineIdx >= allLines.length) return;
      
      const fullLine = allLines[lineIdx];
      
      if (charIdx < fullLine.length) {
        setCurrentText((prev) => prev + fullLine.charAt(charIdx));
        charIdx++;
        typingTimer = setTimeout(typeWriter, 20); 
      } else {
        setCompletedLines((prev) => [...prev, fullLine]);
        setCurrentText(""); 
        lineIdx++; 
        charIdx = 0;
        typingTimer = setTimeout(typeWriter, 300); 
      }
    };
    
    typingTimer = setTimeout(typeWriter, 500);
    return () => clearTimeout(typingTimer);
  }, []);

  return (
    <div className="w-full max-w-4xl bg-[#111] border-2 border-mistral-black p-6 rounded-none shadow-[8px_8px_0px_0px_rgba(250,82,15,1)] text-left relative z-10">
      <div className="flex gap-2 mb-6 border-b border-gray-700 pb-4">
        <div className="w-4 h-4 rounded-full bg-red-500"></div>
        <div className="w-4 h-4 rounded-full bg-mistral-yellow"></div>
        <div className="w-4 h-4 rounded-full bg-green-500"></div>
      </div>
      
      <div className="font-mono text-lg text-mistral-amber flex flex-col gap-2 min-h-[200px]">
        
        {/* CÁC DÒNG ĐÃ GÕ XONG: Thêm animate-pulse vào đây để nó nhấp nháy mãi mãi */}
        {completedLines.map((line, i) => (
          <div key={i} className="animate-pulse">{line}</div>
        ))}
        
        {/* DÒNG ĐANG GÕ VÀ CON TRỎ CHUỘT (Chỉ hiện khi chưa gõ xong hết) */}
        {completedLines.length < allLines.length && (
          <div className="flex items-center">
            <span>{currentText}</span>
            <div className="w-2.5 h-5 bg-mistral-orange animate-pulse ml-1"></div>
          </div>
        )}
        
      </div>
    </div>
  );
}

function TraitBar({ leftLabel, rightLabel, leftValue, rightValue }) {
  return (
    <div className="flex flex-col gap-2 group">
      <div className="flex justify-between text-sm font-bold tracking-wider transition-colors group-hover:text-mistral-orange">
        <span>{leftLabel} ({leftValue}%)</span>
        <span>{rightLabel} ({rightValue}%)</span>
      </div>
      <div className="h-4 w-full bg-gray-200 border border-transparent group-hover:border-mistral-black/10 rounded-none flex overflow-hidden transition-all">
        {/* Thanh màu cam - Thêm hiệu ứng làm sáng khi hover */}
        <div 
          className="h-full bg-gradient-to-r from-mistral-yellow to-mistral-orange transition-all duration-1000 ease-out group-hover:saturate-150" 
          style={{ width: `${leftValue}%` }}
        ></div>
        {/* Thanh màu đen */}
        <div 
          className="h-full bg-mistral-black transition-all duration-1000 ease-out" 
          style={{ width: `${rightValue}%` }}
        ></div>
      </div>
    </div>
  )
}

export default App