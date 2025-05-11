import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom'; // Removed Navigate as no /signin redirect
import { AiOutlineHome, AiOutlineSearch, AiOutlineBook, AiOutlineMenu } from 'react-icons/ai';
import { FaTags, FaBook, FaSearch, FaMicrophone, FaTimes } from 'react-icons/fa'; // Removed Firebase-specific icons
import { Card, CardHeader, CardTitle, CardContent } from './Card'; // Ensure Card.js is compatible or also uses Tailwind
import logo from './logo.jpg';
// NO Firebase imports

// AudioSphere component from your "good looking" version
const AudioSphere = ({ analyser, isRecording }) => {
  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !analyser) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    const resizeCanvas = () => {
      if (canvas.parentElement) { // Check if parentElement is not null
        canvas.width = canvas.parentElement.clientWidth * 0.95;
        canvas.height = Math.min(250, window.innerHeight * 0.3);
      }
    };
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    analyser.fftSize = 2048;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      if (!isRecording) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        return;
      }
      analyser.getByteFrequencyData(dataArray);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const bgGradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
      bgGradient.addColorStop(0, 'rgba(59, 7, 100, 0.2)');
      bgGradient.addColorStop(0.5, 'rgba(147, 51, 234, 0.1)');
      bgGradient.addColorStop(1, 'rgba(236, 72, 153, 0.2)');
      ctx.fillStyle = bgGradient;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      const avgFrequency = dataArray.reduce((sum, value) => sum + value, 0) / bufferLength;
      const baseRadius = Math.min(canvas.width, canvas.height) * 0.25;
      const sphereRadius = baseRadius * (0.8 + (avgFrequency / 255) * 0.3);
      const time = Date.now() * 0.001;
      ctx.save();
      const glowRadius = sphereRadius * 1.2;
      const glowGradient = ctx.createRadialGradient(centerX, centerY, sphereRadius * 0.8, centerX, centerY, glowRadius);
      glowGradient.addColorStop(0, 'rgba(255, 255, 255, 0.3)');
      glowGradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
      ctx.fillStyle = glowGradient;
      ctx.beginPath();
      ctx.arc(centerX, centerY, glowRadius, 0, Math.PI * 2);
      ctx.fill();
      const sphereGradient = ctx.createRadialGradient(centerX - sphereRadius * 0.3, centerY - sphereRadius * 0.3, 0, centerX, centerY, sphereRadius);
      sphereGradient.addColorStop(0, 'rgba(255, 255, 255, 0.95)');
      sphereGradient.addColorStop(0.3, 'rgba(255, 255, 255, 0.7)');
      sphereGradient.addColorStop(0.7, 'rgba(200, 200, 255, 0.5)');
      sphereGradient.addColorStop(1, 'rgba(147, 51, 234, 0.2)');
      ctx.fillStyle = sphereGradient;
      ctx.beginPath();
      ctx.arc(centerX, centerY, sphereRadius, 0, Math.PI * 2);
      ctx.fill();
      const latticeCount = 20;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
      ctx.lineWidth = 0.5;
      for (let i = 0; i < latticeCount; i++) {
        const segment = (i / latticeCount) * Math.PI;
        const latRadius = Math.sin(segment) * sphereRadius;
        const yOffset = Math.cos(segment) * sphereRadius;
        ctx.beginPath();
        for (let angle = 0; angle <= Math.PI * 2; angle += 0.05) {
          const freqIndex = Math.floor((angle / (Math.PI * 2)) * bufferLength / 4);
          const distortion = dataArray[freqIndex] / 255 * 8;
          const waveDistortion = Math.sin(angle * 8 + time * 5) * distortion;
          const radius = latRadius + waveDistortion;
          const x = centerX + Math.cos(angle) * radius;
          const y = centerY + yOffset;
          if (angle === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.stroke();
      }
      for (let i = 0; i < latticeCount; i++) {
        const angle = (i / latticeCount) * Math.PI * 2;
        const xBase = Math.cos(angle);
        // const zBase = Math.sin(angle); // zBase not used in original for x,y plot
        ctx.beginPath();
        for (let segment = 0; segment <= Math.PI; segment += 0.05) {
          const freqIndex = Math.floor((segment / Math.PI) * bufferLength / 4);
          const distortion = dataArray[freqIndex] / 255 * 8;
          const waveDistortion = Math.sin(segment * 8 + time * 5) * distortion;
          const radius = sphereRadius + waveDistortion;
          const latRadius = Math.sin(segment) * radius;
          const yOffset = Math.cos(segment) * radius;
          const x = centerX + xBase * latRadius;
          const y = centerY + yOffset;
          if (segment === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
      const particleCount = 40;
      for (let i = 0; i < particleCount; i++) {
        const freqIndex = Math.floor((i / particleCount) * bufferLength / 4);
        const intensity = dataArray[freqIndex] / 255;
        const angle = (i / particleCount) * Math.PI * 2;
        const orbitRadius = sphereRadius * (1.2 + Math.sin(time * 2 + i * 0.5) * 0.2);
        const x = centerX + Math.cos(angle + time * (0.5 + i * 0.05)) * orbitRadius;
        const y = centerY + Math.sin(angle + time * (0.3 + i * 0.05)) * orbitRadius * 0.5;
        const particleSize = 1 + intensity * 3;
        const particleOpacity = 0.2 + intensity * 0.7;
        ctx.fillStyle = `rgba(255, 255, 255, ${particleOpacity})`;
        ctx.beginPath();
        ctx.arc(x, y, particleSize, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
      ctx.lineWidth = 0.5;
      for (let i = 0; i < particleCount - 1; i++) {
        const angle1 = (i / particleCount) * Math.PI * 2;
        const orbitRadius1 = sphereRadius * (1.2 + Math.sin(time * 2 + i * 0.5) * 0.2);
        const x1 = centerX + Math.cos(angle1 + time * (0.5 + i * 0.05)) * orbitRadius1;
        const y1 = centerY + Math.sin(angle1 + time * (0.3 + i * 0.05)) * orbitRadius1 * 0.5;
        for (let j = i + 1; j < Math.min(i + 3, particleCount); j++) {
          const angle2 = (j / particleCount) * Math.PI * 2;
          const orbitRadius2 = sphereRadius * (1.2 + Math.sin(time * 2 + j * 0.5) * 0.2);
          const x2 = centerX + Math.cos(angle2 + time * (0.5 + j * 0.05)) * orbitRadius2;
          const y2 = centerY + Math.sin(angle2 + time * (0.3 + j * 0.05)) * orbitRadius2 * 0.5;
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();
        }
      }
      ctx.restore();
      animationFrameRef.current = requestAnimationFrame(draw);
    };

    if (isRecording) draw();
    return () => {
      window.removeEventListener('resize', resizeCanvas);
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, [analyser, isRecording]);

  // Tailwind classes from your "good looking" version
  return (
    <div className="relative overflow-hidden bg-gradient-to-b from-indigo-900/20 to-purple-900/20 rounded-b-2xl shadow-inner">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(255,255,255,0.15),transparent)] animate-pulse-slow" />
      <canvas 
        ref={canvasRef} 
        className="w-full max-w-[1000px] mx-auto relative z-10"
      />
    </div>
  );
};

// This SearchBar component should be the one from your "good looking" original version.
// It will be defined *inside* the Dashboard component later or passed props.
// For now, assume it's defined globally for structure, then we'll integrate.
const OriginalSearchBar = ({ isRecordingProp, analyserProp, startRecordingFunc, stopRecordingFunc, setSearchFocusFunc }) => {
  // This component needs access to isRecording, analyser, startRecording, stopRecording from the Dashboard state
  // We'll pass them as props.
  const [searchFocus, setSearchFocusLocal] = useState(false); // Local focus for input

  const handleFocus = () => {
    setSearchFocusLocal(true);
    if (setSearchFocusFunc) setSearchFocusFunc(true);
  }
  const handleBlur = () => {
    setSearchFocusLocal(false);
    if (setSearchFocusFunc) setSearchFocusFunc(false);
  }

  return (
    <div className="relative group z-20">
      <div className={`absolute inset-0 bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 rounded-2xl transition-all duration-500 ${isRecordingProp ? 'opacity-90 blur-xl scale-105' : 'opacity-70 blur-md scale-100'}`} />
      <div className="relative bg-white/95 backdrop-blur-sm rounded-2xl shadow-xl overflow-hidden border border-blue-100">
        <div className="flex items-center p-4">
          <FaSearch className="text-indigo-500 text-xl mr-4 flex-shrink-0" />
          <input
            type="text"
            placeholder="Ask anything with voice or text..."
            className="w-full text-lg text-gray-800 placeholder-gray-400 bg-transparent border-none outline-none transition-all duration-300"
            onFocus={handleFocus}
            onBlur={handleBlur}
          />
          <button
            onClick={() => {
              if (isRecordingProp) stopRecordingFunc();
              else startRecordingFunc();
            }}
            className={`ml-4 p-3 rounded-full transition-all duration-500 relative overflow-hidden group/button ${isRecordingProp ? 'bg-indigo-500 text-white scale-110' : 'text-indigo-500 hover:bg-indigo-50'}`}
          >
            <span className={`absolute inset-0 bg-gradient-to-r from-purple-500/20 to-indigo-500/40 scale-0 group-hover/button:scale-150 origin-center transition-transform duration-500 ${isRecordingProp ? 'scale-150' : ''}`} />
            <FaMicrophone className={`text-xl relative z-10 ${isRecordingProp ? 'animate-pulse' : ''}`} />
          </button>
        </div>
        {isRecordingProp && analyserProp && ( // Check analyserProp as well
          <div className="px-4 pb-4">
            <AudioSphere analyser={analyserProp} isRecording={isRecordingProp} />
          </div>
        )}
      </div>
    </div>
  );
};


const Dashboard = () => {
  const navigate = useNavigate();
  // const location = useLocation(); // Not used if no Firebase

  const [activeTab, setActiveTab] = useState('home');
  // const [userName, setUserName] = useState('Guest'); // Simplified
  const [searchFocus, setSearchFocus] = useState(false); // For overall page effect if needed
  const [isRecording, setIsRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [isSidebarOpen, setSidebarOpen] = useState(true); // Default to true for desktop
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
  // const [loading, setLoading] = useState(false); // No Firebase loading
  // const [error, setError] = useState(null); // No Firebase error

  const [audioContext, setAudioContext] = useState(null);
  const [analyser, setAnalyser] = useState(null);
  const [mediaStream, setMediaStream] = useState(null);

  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      setSidebarOpen(!mobile); // Auto-close on mobile, auto-open on desktop
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const toggleSidebar = () => setSidebarOpen(!isSidebarOpen);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const context = new (window.AudioContext || window.webkitAudioContext)();
      const audioAnalyser = context.createAnalyser();
      const source = context.createMediaStreamSource(stream);
      
      source.connect(audioAnalyser);
      setAudioContext(context);
      setAnalyser(audioAnalyser);
      setMediaStream(stream);
      // setIsRecording(true); // Set after MediaRecorder setup to ensure flow

      const recorder = new MediaRecorder(stream);
      recorder.ondataavailable = async (event) => {
        if (event.data.size > 0) {
          const formData = new FormData();
          formData.append("audio", event.data, "voice-input.wav");
          try {
            console.log("Sending audio to /api/voice-input/");
            // const response = await fetch("/api/voice-input/", { method: "POST", body: formData });
            // if (response.ok) {
            //   const result = await response.json();
            //   console.log("API Response:", result);
            // } else {
            //   console.error("API Error:", response.statusText);
            // }
          } catch (fetchError) {
            console.error("Fetch error:", fetchError);
          }
        }
      };
      recorder.onstart = () => {
        console.log("Recorder started");
        setIsRecording(true); // Now set isRecording, UI updates
      }
      recorder.onstop = () => {
        console.log("Recorder stopped");
        // setIsRecording(false); // Handled by main stopRecording function
      };
      recorder.onerror = (e) => {
        console.error("MediaRecorder error:", e);
        setIsRecording(false); // Reset state on error
      }
      recorder.start();
      setMediaRecorder(recorder);
    } catch (error) {
      console.error("Error accessing microphone:", error);
      setIsRecording(false); // Reset state on error
    }
  };

  const stopRecording = () => {
    if (mediaRecorder && mediaRecorder.state === "recording") {
      mediaRecorder.stop();
    }
    if (mediaStream) {
      mediaStream.getTracks().forEach(track => track.stop());
    }
    if (audioContext && audioContext.state !== 'closed') {
      audioContext.close().catch(e => console.error("Error closing AudioContext:", e));
    }
    setAudioContext(null);
    setAnalyser(null); // Important to clear analyser for AudioSphere
    setMediaStream(null);
    setMediaRecorder(null);
    setIsRecording(false);
  };

  const navItems = [
    { id: 'home', icon: AiOutlineHome, label: 'Home' },
    { id: 'search', icon: AiOutlineSearch, label: 'Search' },
    { id: 'library', icon: FaBook, label: 'Library' },
    { id: 'documentation', icon: AiOutlineBook, label: 'Documentation' },
    { id: 'billing', icon: FaTags, label: 'Billing' }, // Renamed from Platform Info for consistency
  ];

  // Sidebar using Tailwind classes from your "good looking" version
  const Sidebar = () => (
    <div className={`
      fixed inset-y-0 left-0 z-30 w-64 bg-white border-r border-indigo-100 shadow-lg
      transform transition-transform duration-300 ease-in-out
      ${isMobile ? (isSidebarOpen ? 'translate-x-0' : '-translate-x-full') : (isSidebarOpen ? 'translate-x-0' : 'translate-x-0')} 
      flex flex-col
    `}> {/* Ensured sidebar is always open on desktop if isSidebarOpen is true */}
      <div className="p-4 border-b border-indigo-100 flex justify-between items-center">
        <div className="flex items-center space-x-3 cursor-pointer" onClick={() => navigate('/')}>
          <img src={logo} alt="ConvoLabs Logo" className="w-8 h-8" />
          <span className="text-xl font-bold text-indigo-600">VoiceEd India</span> {/* Updated name */}
        </div>
        {isMobile && (
          <button onClick={toggleSidebar} className="text-gray-500 hover:text-gray-700">
            <FaTimes size={24} />
          </button>
        )}
      </div>
      <nav className="p-4 space-y-2 flex-grow">
        {navItems.map((item) => (
          <button
            key={item.id}
            onClick={() => {
              setActiveTab(item.id);
              if (isMobile) setSidebarOpen(false);
            }}
            className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg ${activeTab === item.id ? 'bg-indigo-50 text-indigo-600' : 'text-gray-600 hover:bg-gray-50'}`}
          >
            <item.icon size={20} />
            <span>{item.label}</span>
          </button>
        ))}
      </nav>
      <div className="p-4 border-t border-indigo-100">
        {/* Simplified footer, no user/logout */}
        <p className="text-xs text-gray-400 text-center">© VoiceEd India</p>
      </div>
    </div>
  );

  // Content components using Tailwind classes from your "good looking" version
  const contentComponents = {
    home: (
      <div className="min-h-screen bg-gray-50">
        <div className="relative bg-gray-50 -mt-30"> {/* Your original negative margin */}
          <div className="container mx-auto px-4 space-y-6 pb-12">
            <Card className="bg-white shadow-lg border border-gray-100">
              <CardHeader className="border-b border-gray-100">
                {/* Simplified title */}
                <CardTitle className="text-2xl text-gray-800">Welcome to VoiceEd India!</CardTitle>
              </CardHeader>
            </Card>
          </div>
        </div>
        <div className="relative h-[60vh] flex flex-col items-center justify-center"> {/* Original centering */}
          <div className={`max-w-3xl w-full transition-all duration-300 ${searchFocus ? "scale-105" : ""}`}>
            <OriginalSearchBar
              isRecordingProp={isRecording}
              analyserProp={analyser}
              startRecordingFunc={startRecording}
              stopRecordingFunc={stopRecording}
              setSearchFocusFunc={setSearchFocus} // To control overall page focus if needed
            />
          </div>
        </div>
      </div>
    ),
    // Placeholder for other tabs, you'd style these with Tailwind too
    search: (
      <Card className="bg-white shadow-lg border border-gray-100 m-4">
        <CardHeader><CardTitle className="text-xl text-gray-800">Search</CardTitle></CardHeader>
        <CardContent><p className="text-gray-600 p-4">Search content here...</p></CardContent>
      </Card>
    ),
    library: (
      <Card className="bg-white shadow-lg border border-gray-100 m-4">
        <CardHeader><CardTitle className="text-xl text-gray-800">Library</CardTitle></CardHeader>
        <CardContent><p className="text-gray-600 p-4">Library content here...</p></CardContent>
      </Card>
    ),
    documentation: (
      <Card className="bg-white shadow-lg border border-gray-100 m-4">
        <CardHeader><CardTitle className="text-xl text-gray-800">Documentation</CardTitle></CardHeader>
        <CardContent><p className="text-gray-600 p-4">Documentation content here...</p></CardContent>
      </Card>
    ),
    billing: (
      <Card className="bg-white shadow-lg border border-gray-100 m-4">
        <CardHeader><CardTitle className="text-xl text-gray-800">Billing</CardTitle></CardHeader>
        <CardContent><p className="text-gray-600 p-4">Billing content here...</p></CardContent>
      </Card>
    ),
  };

  // Main layout using Tailwind classes from your "good looking" version
  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 to-blue-50 flex">
      {/* Mobile Header */}
      {isMobile && (
        <div className="fixed top-0 left-0 right-0 h-16 bg-white border-b border-indigo-100 flex items-center justify-between px-4 z-40">
          <button onClick={toggleSidebar} className="text-gray-500 hover:text-gray-700">
            <AiOutlineMenu size={24} />
          </button>
          <div className="flex items-center space-x-3">
            <img src={logo} alt="ConvoLabs Logo" className="w-8 h-8" />
            <span className="text-xl font-bold text-indigo-600">VoiceEd</span> {/* Shortened for mobile */}
          </div>
          <div className="w-8" /> {/* Spacer */}
        </div>
      )}
      {/* Overlay for mobile sidebar */}
      {isMobile && isSidebarOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-20" onClick={toggleSidebar} />
      )}
      {/* Main Layout */}
      <div className="flex w-full">
        <Sidebar />
        <main className="flex-1 overflow-y-auto" style={{ marginLeft: isMobile ? '0' : (isSidebarOpen ? '16rem' : '0') }}> {/* Adjust margin based on sidebar state for desktop */}
          <div className={`${isMobile ? 'pt-20 px-4' : 'p-8'} min-h-screen`}> {/* More padding top for mobile if header is fixed */}
            {contentComponents[activeTab]}
          </div>
        </main>
      </div>
      {/* ProfileModal removed */}
    </div>
  );
};

export default Dashboard;