// Eastern Mediterranean map — Leaflet with a clean dark basemap (CartoDB Dark Matter).
// Drops custom DOM markers for the 3 monitored basins; click to focus.

const { useEffect, useRef } = React;

const RegionMap = ({ active, onSelect, regions, summaries }) => {
  const mapEl = useRef(null);
  const mapRef = useRef(null);
  const layersRef = useRef({});

  // Initialize map once.
  useEffect(() => {
    if (!mapEl.current || mapRef.current) return;

    const map = L.map(mapEl.current, {
      center: [37.0, 28.0],
      zoom: 6,
      minZoom: 5,
      maxZoom: 9,
      zoomControl: false,
      scrollWheelZoom: false,
      attributionControl: true,
      worldCopyJump: false,
      preferCanvas: true,
    });
    mapRef.current = map;

    // Detect theme — pick the right basemap variant.
    const isLight = document.documentElement.classList.contains('theme-light');
    const url = isLight
      ? 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}'
      : 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}';
    const labelsUrl = isLight
      ? 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}'
      : 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}';

    const base = L.tileLayer(url, {
      maxZoom: 16,
      attribution: 'Tiles &copy; Esri — Esri, DeLorme, NAVTEQ',
    }).addTo(map);
    const labels = L.tileLayer(labelsUrl, {
      maxZoom: 16,
      pane: 'overlayPane',
    }).addTo(map);

    layersRef.current.base = base;
    layersRef.current.labels = labels;

    // Add a custom control card overlay with title + scale.
    const titleCtrl = L.control({ position: 'bottomleft' });
    titleCtrl.onAdd = function() {
      const div = L.DomUtil.create('div', 'mars-map-title');
      div.innerHTML = `
        <div class="mars-map-title-eyebrow">EASTERN MEDITERRANEAN · 19°–36° E</div>
        <div class="mars-map-title-text">3 monitored basins</div>
      `;
      return div;
    };
    titleCtrl.addTo(map);

    // Custom zoom buttons
    const zoomCtrl = L.control({ position: 'topright' });
    zoomCtrl.onAdd = function() {
      const wrap = L.DomUtil.create('div', 'mars-map-zoom');
      wrap.innerHTML = `
        <button data-z="+" title="Zoom in" type="button">+</button>
        <button data-z="-" title="Zoom out" type="button">−</button>
        <button data-z="fit" title="Reset view" type="button">⤢</button>
      `;
      L.DomEvent.disableClickPropagation(wrap);
      wrap.querySelector('[data-z="+"]').onclick = () => map.zoomIn();
      wrap.querySelector('[data-z="-"]').onclick = () => map.zoomOut();
      wrap.querySelector('[data-z="fit"]').onclick = () => map.setView([37.0, 28.0], 6);
      return wrap;
    };
    zoomCtrl.addTo(map);

    // Region pins (markers + bbox rectangles).
    const pinCoords = {
      thermaikos: [(40.2+40.7)/2, (22.5+23.0)/2],
      peiraeus:   [(37.9+38.1)/2, (23.5+23.8)/2],
      limassol:   [(34.6+34.8)/2, (33.0+33.2)/2],
    };

    Object.entries(regions).forEach(([k, v]) => {
      const sum = summaries[k];
      const level = sum?.level || 'low';
      const colorVar = level === 'high' ? '--danger' : level === 'med' ? '--warn' : '--ok';
      const color = getComputedStyle(document.documentElement).getPropertyValue(colorVar).trim();

      // BBox outline
      const [latMin, latMax, lonMin, lonMax] = v.bbox;
      const rect = L.rectangle([[latMin, lonMin], [latMax, lonMax]], {
        color: color,
        weight: 1.2,
        opacity: 0.6,
        fillColor: color,
        fillOpacity: 0.10,
        interactive: true,
      }).addTo(map);
      rect.on('click', () => onSelect(k));

      // Pin (custom DOM)
      const html = `
        <div class="mars-pin" data-level="${level}" data-key="${k}">
          <div class="mars-pin-ring"></div>
          <div class="mars-pin-dot"></div>
          <div class="mars-pin-label">
            <div class="mars-pin-label-name">${v.short || v.title.split(' (')[0]}</div>
            <div class="mars-pin-label-meta">CHL ${sum?.chl?.toFixed(3) ?? '—'} mg/m³</div>
          </div>
        </div>
      `;
      const icon = L.divIcon({
        className: 'mars-pin-icon',
        html,
        iconSize: [0, 0],
        iconAnchor: [0, 0],
      });
      const marker = L.marker(pinCoords[k], { icon, riseOnHover: true });
      marker.on('click', () => onSelect(k));
      marker.addTo(map);

      layersRef.current[k] = { rect, marker };
    });

    // Lock interactions to keep the basins always in view.
    map.setMaxBounds(L.latLngBounds([29, 14], [46, 40]));
  }, []);

  // Update active region styling.
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    Object.entries(regions).forEach(([k, v]) => {
      const obj = layersRef.current[k];
      if (!obj) return;
      const sum = summaries[k];
      const level = sum?.level || 'low';
      const colorVar = level === 'high' ? '--danger' : level === 'med' ? '--warn' : '--ok';
      const color = getComputedStyle(document.documentElement).getPropertyValue(colorVar).trim();
      const isActive = k === active;
      obj.rect.setStyle({
        color, fillColor: color,
        weight: isActive ? 2 : 1.2,
        opacity: isActive ? 0.9 : 0.5,
        fillOpacity: isActive ? 0.20 : 0.07,
        dashArray: isActive ? null : '4 4',
      });
      const el = obj.marker.getElement();
      if (el) {
        const pin = el.querySelector('.mars-pin');
        if (pin) {
          pin.classList.toggle('active', isActive);
          pin.dataset.level = level;
        }
      }
    });
  }, [active, summaries]);

  // Swap tile layer if theme changes.
  useEffect(() => {
    const obs = new MutationObserver(() => {
      const map = mapRef.current;
      if (!map || !layersRef.current.base) return;
      const isLight = document.documentElement.classList.contains('theme-light');
      const url = isLight
        ? 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}'
        : 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}';
      const labelsUrl = isLight
        ? 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}'
        : 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}';
      layersRef.current.base.setUrl(url);
      layersRef.current.labels.setUrl(labelsUrl);
    });
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    return () => obs.disconnect();
  }, []);

  return <div ref={mapEl} className="mars-leaflet" />;
};

window.RegionMap = RegionMap;
