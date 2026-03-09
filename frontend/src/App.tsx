import { useState, useEffect, useRef } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { SystemClock } from './components/SystemClock';
import { TracePanel } from './components/TracePanel';
import { PostList } from './components/PostList';
import { PostDetail } from './pages/PostDetail';
import { getPosts, getTraces, getTime } from './lib/api';
import type { Post, Trace } from './types';

function Home({ posts, systemTime, onLoadMore, hasMore, isLoadingMore }: { posts: Post[]; systemTime: number; onLoadMore: () => void; hasMore: boolean; isLoadingMore: boolean }) {
  return (
    <div className="max-w-2xl mx-auto pt-24 pb-20 px-4">
      <PostList posts={posts} systemTime={systemTime} onLoadMore={onLoadMore} hasMore={hasMore} isLoadingMore={isLoadingMore} />
    </div>
  );
}

function App() {
  const [posts, setPosts] = useState<Post[]>([]);
  const [traces, setTraces] = useState<Trace[]>([]);
  const [isPausedTraces, setIsPausedTraces] = useState(false);
  const [, setErrorCount] = useState(0);
  const [pollInterval, setPollInterval] = useState(3000); // Changed to 3s
  const [systemTime, setSystemTime] = useState<number>(Date.now());
  const [hasMore, setHasMore] = useState(true);
  const [isLoadingMore, setIsLoadingMore] = useState(false);

  const isPollingRef = useRef(false);

  const handleLoadMore = async () => {
    if (isLoadingMore || !hasMore) return;
    setIsLoadingMore(true);
    try {
      const morePosts = await getPosts(20, posts.length);
      if (morePosts.length < 20) {
        setHasMore(false);
      }
      setPosts(prev => [...prev, ...morePosts]);
    } catch (error) {
      console.error("Failed to load more posts", error);
    } finally {
      setIsLoadingMore(false);
    }
  };

  useEffect(() => {
    const fetchData = async () => {
      if (isPollingRef.current) return;
      isPollingRef.current = true;

      try {
        // Concurrent requests
        // For posts polling, we only want to fetch new posts to prepend.
        // But our API doesn't support "since_id".
        // So we fetch the latest page (offset=0) and merge.
        // If we have loaded multiple pages (e.g. 100 posts), fetching just 20 might create a gap if we replaced.
        // But here we will just merge the NEW ones at the top.
        const [newPosts, newTraces, currentTimeData] = await Promise.all([
          getPosts(20, 0), // Poll latest 20
          !isPausedTraces ? getTraces(50) : Promise.resolve([]),
          getTime()
        ]);

        // Use system time for relative time calculation
        // The backend returns current_time as a string (YYYY-MM-DD HH:mm:ss.SSSSSS) or similar.
        const systemTime = new Date(currentTimeData.current_time).getTime();

        setPosts(prev => {
          // Merge logic for posts:
          // 1. Identify new posts (that are not in prev)
          // 2. Prepend them
          // 3. Update existing posts (stats might have changed) - optional but good.
          // Since we don't want to replace the whole list (which might be long due to Load More),
          // we only update the ones we fetched.

          if (newPosts.length === 0) return prev;

          const prevIds = new Set(prev.map(p => p.id));
          const trulyNewPosts = newPosts.filter(p => !prevIds.has(p.id));

          // Also update stats for overlapping posts?
          // For simplicity, let's just prepend new ones.
          // If we want to update stats of visible posts, we'd need to map over prev.
          // Let's do a smart merge:
          // Create a map of newPosts by ID.
          const newPostsMap = new Map(newPosts.map(p => [p.id, p]));

          const updatedPrev = prev.map(p => newPostsMap.has(p.id) ? newPostsMap.get(p.id)! : p);

          return [...trulyNewPosts, ...updatedPrev];
        });

        // We need to store systemTime in state to pass it down
        setSystemTime(systemTime);

        if (!isPausedTraces && Array.isArray(newTraces)) {
          setTraces(prev => {
            // If newTraces is empty, don't change anything
            if (newTraces.length === 0) return prev;

            const existingIds = new Set(prev.map(t => t.id));
            const uniqueNew = newTraces.filter(t => !existingIds.has(t.id));

            if (uniqueNew.length === 0) return prev;

            // Prepend new traces to the list
            let updated = [...uniqueNew, ...prev];
            if (updated.length > 200) updated = updated.slice(0, 150);
            return updated;
          });
        }

        // Success - reset error count
        setErrorCount(0);
        if (pollInterval !== 3000) setPollInterval(3000);

      } catch (error) {
        console.error("Polling error", error);
        setErrorCount(prev => {
          const newCount = prev + 1;
          if (newCount >= 3) setPollInterval(15000);
          return newCount;
        });
      } finally {
        isPollingRef.current = false;
      }
    };

    fetchData(); // Fetch immediately on mount
    const id = setInterval(fetchData, pollInterval);
    return () => clearInterval(id);
  }, [pollInterval, isPausedTraces]); // Re-run when interval changes

  return (
    <BrowserRouter>
      <div className="min-h-screen bg-slate-50 text-slate-800 font-body">
        <SystemClock />
        <TracePanel
          traces={traces}
          isPaused={isPausedTraces}
          onPauseToggle={() => setIsPausedTraces(!isPausedTraces)}
          onClear={() => setTraces([])}
        />

        <Routes>
          <Route path="/" element={<Home posts={posts} systemTime={systemTime} onLoadMore={handleLoadMore} hasMore={hasMore} isLoadingMore={isLoadingMore} />} />
          <Route path="/post/:id" element={<PostDetail />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
