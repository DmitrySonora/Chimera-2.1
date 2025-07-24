import pytest
from actors.user_session_actor import UserSessionActor, UserSession
from config.logging import setup_logging

# Настраиваем логирование для тестов
setup_logging()


class TestModeDetection:
    """Тесты для определения режимов общения"""
    
    @pytest.mark.asyncio
    async def test_expert_mode_detection(self):
        """Тест определения expert режима"""
        actor = UserSessionActor()
        session = UserSession(user_id="test_user")
        
        # Вопросы должны определяться как expert
        test_cases = [
            "Объясни, как работает нейронная сеть?",
            "Почему небо голубое?",
            "Расскажи про принцип работы двигателя",
            "Что такое квантовая механика?"
        ]
        
        for text in test_cases:
            mode, confidence = actor._determine_generation_mode(text, session)
            assert mode == 'expert', f"Text '{text}' should be expert mode"
            assert confidence > 0.5, f"Confidence should be > 0.5, got {confidence}"
    
    @pytest.mark.asyncio
    async def test_creative_mode_detection(self):
        """Тест определения creative режима"""
        actor = UserSessionActor()
        session = UserSession(user_id="test_user")
        
        test_cases = [
            "Придумай историю про дракона",
            "Сочини сказку для детей",
            "Представь, что ты космонавт",
            "Выдумай фантастический мир"
        ]
        
        for text in test_cases:
            mode, confidence = actor._determine_generation_mode(text, session)
            assert mode == 'creative', f"Text '{text}' should be creative mode"
            assert confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_talk_mode_default(self):
        """Тест режима talk по умолчанию"""
        actor = UserSessionActor()
        session = UserSession(user_id="test_user")
        
        # Обычные сообщения
        test_cases = [
            "Привет, как дела?",
            "Мне сегодня грустно",
            "Какое у тебя настроение?",
            "Что-то скучно стало"
        ]
        
        for text in test_cases:
            mode, confidence = actor._determine_generation_mode(text, session)
            assert mode == 'talk', f"Text '{text}' should be talk mode"
    
    @pytest.mark.asyncio
    async def test_short_text_handling(self):
        """Тест обработки коротких сообщений"""
        actor = UserSessionActor()
        session = UserSession(user_id="test_user")
        
        # Установим текущий режим
        session.current_mode = 'expert'
        
        # Короткий текст должен использовать текущий режим
        mode, confidence = actor._determine_generation_mode("Да", session)
        assert mode == 'expert'
        assert confidence == 0.5
        
        # Пустой текст
        mode, confidence = actor._determine_generation_mode("", session)
        assert mode == 'expert'
        assert confidence == 0.5
    
    @pytest.mark.asyncio
    async def test_mode_history(self):
        """Тест сохранения истории режимов"""
        actor = UserSessionActor()
        session = UserSession(user_id="test_user")
        
        # Добавляем историю одинаковых режимов
        session.mode_history = ['expert', 'expert', 'expert']
        
        # При определении того же режима уверенность должна возрасти
        mode, confidence = actor._determine_generation_mode(
            "Объясни квантовую физику", 
            session
        )
        assert mode == 'expert'
        assert confidence > 0.7  # Усиленная уверенность
    
    @pytest.mark.asyncio
    async def test_mixed_patterns(self):
        """Тест смешанных паттернов"""
        actor = UserSessionActor()
        session = UserSession(user_id="test_user")
        
        # Текст с паттернами разных режимов
        text = "Придумай научное объяснение почему драконы дышат огнем"
        mode, confidence = actor._determine_generation_mode(text, session)
        
        # Должен выбрать creative из-за более высокого веса
        assert mode in ['creative', 'expert']  # Оба валидны
        assert confidence > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])