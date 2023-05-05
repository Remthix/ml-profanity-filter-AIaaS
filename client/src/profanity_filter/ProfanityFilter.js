import React, { useState } from 'react';
import axios from 'axios';

const API_ENDOINT = 'http://127.0.0.1:5000/check';

const ProfanityFilter = () => {
    const [inputText, setInputText] = useState('');
    const [hasProfanity, setHasProfanity] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const checkProfanity = async (text) => {
        try {
            setIsLoading(true);
            const response = await axios.post(API_ENDOINT, {
                text: text,
            });
            setHasProfanity(JSON.parse(response.data.contains_profanity));
        } catch (err) {
            setError('Problem executing API call!');
        } finally {
            setIsLoading(false);
        }
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        checkProfanity(inputText);
    };

    return (
        <div>
            <h1>Profanity Filter</h1>
            <form onSubmit={handleSubmit}>
                <div>
                    <label htmlFor="input-text">Enter text to be checked:</label>
                    <textarea
                        id="input-text"
                        value={inputText}
                        onChange={(e) => setInputText(e.target.value)}
                    />
                </div>
                <button type="submit">Check Profanity</button>
            </form>

            {isLoading && <p>Loading...</p>}
            {error && <p>{error}</p>}
            {hasProfanity !== null && (
                <p>
                    {hasProfanity
                        ? 'Text contains profanity!'
                        : 'Text does not contain profanity.'}
                </p>
            )}

        </div>
    );
};

export default ProfanityFilter;